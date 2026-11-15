// Sharded descriptor heap state and the sparse descriptor store.
// Split from DX12Tracking.h; keep declarations in the layer that owns them.
#pragma once

#include "DX12FrameAnalysisTracking.h"

#include <algorithm>
#include <bit>
#include <utility>

// Lock and version storage for one descriptor heap. Each shard occupies its
// own cache line: packing SRWLOCKs or counters together would make unrelated
// descriptor slots contend through false sharing even though their logical
// locks are independent.
template <size_t ShardCountValue>
class DX12DescriptorHeapShardState
{
public:
	static constexpr size_t ShardCount = ShardCountValue;
	static constexpr size_t CacheLineBytes = 64;

	static_assert(ShardCount != 0 && (ShardCount & (ShardCount - 1)) == 0,
		"descriptor shard count must be a power of two");

	struct alignas(CacheLineBytes) LockCell
	{
		SRWLOCK value = SRWLOCK_INIT;
	};

	static_assert(sizeof(LockCell) % CacheLineBytes == 0,
		"descriptor locks must not share cache lines");

	explicit DX12DescriptorHeapShardState(uint32_t maxSlots = 0)
		: maxSlots_(maxSlots),
		  versionPageCount_(maxSlots == 0 ? 0 : static_cast<uint32_t>(
			(static_cast<uint64_t>(maxSlots) + VersionPageSize - 1) /
			VersionPageSize))
	{
		for (LockCell &cell : locks_)
			InitializeSRWLock(&cell.value);
		if (versionPageCount_ == 0)
			return;
		versionPages_ =
			std::make_unique<std::atomic<VersionPage*>[]>(versionPageCount_);
		for (uint32_t index = 0; index < versionPageCount_; ++index)
			versionPages_[index].store(nullptr, std::memory_order_relaxed);
	}

	~DX12DescriptorHeapShardState()
	{
		if (!versionPages_)
			return;
		for (uint32_t index = 0; index < versionPageCount_; ++index)
			delete versionPages_[index].load(std::memory_order_relaxed);
	}

	DX12DescriptorHeapShardState(const DX12DescriptorHeapShardState&) = delete;
	DX12DescriptorHeapShardState &operator=(
		const DX12DescriptorHeapShardState&) = delete;

	static constexpr size_t ShardForSlot(uint32_t slot)
	{
		return static_cast<size_t>(slot) & (ShardCount - 1);
	}

	SRWLOCK *LockForSlot(uint32_t slot)
	{
		return &locks_[ShardForSlot(slot)].value;
	}

	void LockAllExclusive()
	{
		for (LockCell &cell : locks_)
			AcquireSRWLockExclusive(&cell.value);
	}

	void UnlockAllExclusive()
	{
		for (size_t i = ShardCount; i > 0; --i)
			ReleaseSRWLockExclusive(&locks_[i - 1].value);
	}

	bool IsActive() const
	{
		return active_.load(std::memory_order_acquire);
	}

	void Deactivate()
	{
		active_.store(false, std::memory_order_release);
	}

	// Dense slot arrays are materialized once per heap lifetime and then
	// stable; a release store here publishes that readiness so lock-free
	// writers/readers can touch slots/occupied without the shard locks.
	bool IsMaterialized() const
	{
		return materialized_.load(std::memory_order_acquire);
	}

	void MarkMaterialized()
	{
		materialized_.store(true, std::memory_order_release);
	}

	const SRWLOCK *LockForSlot(uint32_t slot) const
	{
		return &locks_[ShardForSlot(slot)].value;
	}

	void ResetVersions()
	{
		version_.store(1, std::memory_order_release);
		if (!versionPages_)
			return;
		for (uint32_t pageIndex = 0;
				pageIndex < versionPageCount_; ++pageIndex) {
			VersionPage *page = versionPages_[pageIndex].load(
				std::memory_order_acquire);
			if (!page)
				continue;
			for (typename VersionPage::SlotState &slot : page->slots) {
				slot.version.store(2, std::memory_order_release);
				slot.valueId.store(0, std::memory_order_release);
				slot.flatValueId.store(0, std::memory_order_release);
				slot.flatVersion.store(2, std::memory_order_release);
			}
		}
	}

	void BumpVersion(uint32_t slot)
	{
		PublishSlotValue(slot, 0);
	}

	void PublishSlotValue(uint32_t slot, uint64_t valueId)
	{
		PublishSlotIdentity(slot, valueId);
		BumpHeapVersion();
	}

	// Range copies publish every slot identity, but consumers only require the
	// heap-wide invalidation token to change once for the completed API call.
	// Splitting these operations removes one contended atomic RMW per copied
	// descriptor without weakening per-slot identity validation.
	void PublishSlotIdentity(uint32_t slot, uint64_t valueId)
	{
		VersionPage *page = EnsureVersionPage(slot);
		if (page) {
			typename VersionPage::SlotState &state =
				page->slots[slot & (VersionPageSize - 1)];
			// Odd means a publication is in progress. Readers only accept the
			// value when the same even version brackets the Value ID load. Slot
			// writers are serialized by the owning shard lock, so the second
			// contended atomic RMW formerly used here bought no correctness. An
			// exchange starts publication and the release store completes it.
			const uint64_t previous = state.version.fetch_or(
				1ull, std::memory_order_acq_rel);
			state.valueId.store(valueId, std::memory_order_relaxed);
			// Versions are initialized even and only one writer owns this slot.
			// Advancing from the previous even generation preserves cache-key
			// invalidation, including wraparound's even parity.
			const uint64_t completedVersion = (previous + 2ull) & ~1ull;
			state.version.store(completedVersion,
				std::memory_order_release);
			// Flat snapshot: written after the version store completes so a
			// reader that observes flatVersion == version also observes the
			// matching flatValueId (same thread store order; relaxed loads
			// on x86 observe stores in order).
			state.flatValueId.store(valueId, std::memory_order_relaxed);
			state.flatVersion.store(
				completedVersion, std::memory_order_relaxed);
		}
	}

	// Lock-free O(1) check: when the slot's version is unchanged since the
	// last publication, flatValueIds[slot] is the authoritative valueId.
	// Returns false when the slot was never published or has been republished
	// (the caller then falls back to the seqlock path).
	bool ReadFlatValueId(uint32_t slot, uint64_t *valueId) const
	{
		if (!valueId || !versionPages_ || slot >= maxSlots_)
			return false;
		VersionPage *page = versionPages_[slot / VersionPageSize].load(
			std::memory_order_acquire);
		if (!page)
			return false;
		const typename VersionPage::SlotState &state =
			page->slots[slot & (VersionPageSize - 1)];
		const uint64_t flatVersion =
			state.flatVersion.load(std::memory_order_relaxed);
		if (flatVersion == 0)
			return false;
		if (state.version.load(std::memory_order_acquire) !=
			flatVersion)
			return false;
		*valueId =
			state.flatValueId.load(std::memory_order_relaxed);
		return true;
	}

	void BumpHeapVersion()
	{
		version_.fetch_add(1, std::memory_order_release);
	}

	struct SlotIdentity
	{
		uint64_t version = 0;
		uint64_t valueId = 0;
	};

	bool ObserveSlotIdentity(uint32_t slot, SlotIdentity *identity) const
	{
		if (!identity || !versionPages_ || slot >= maxSlots_)
			return false;
		VersionPage *page = versionPages_[slot / VersionPageSize].load(
			std::memory_order_acquire);
		if (!page) {
			identity->version = 2;
			identity->valueId = 0;
			return true;
		}
		const typename VersionPage::SlotState &state =
			page->slots[slot & (VersionPageSize - 1)];
		const uint64_t before = state.version.load(
			std::memory_order_acquire);
		if ((before & 1) != 0)
			return false;
		const uint64_t valueId = state.valueId.load(
			std::memory_order_acquire);
		const uint64_t after = state.version.load(
			std::memory_order_acquire);
		if (before != after || (after & 1) != 0)
			return false;
		identity->version = after;
		identity->valueId = valueId;
		return true;
	}

	// Creation hooks frequently publish the same compact descriptor value to the
	// same slot thousands of times per frame.  Reject those writes from the
	// seqlock publication itself before taking the destination shard exclusively
	// or touching the sparse descriptor store.
	bool PublishedSlotValueEquals(uint32_t slot, uint64_t valueId) const
	{
		if (valueId == 0)
			return false;
		// Flat snapshot fast path: the post-publication snapshot is
		// authoritative whenever the live version still matches it - the
		// common steady-state case for repeatedly re-created descriptors.
		// Two relaxed loads instead of the seqlock's three-load protocol.
		UINT64 flatValueId = 0;
		if (ReadFlatValueId(slot, &flatValueId))
			return flatValueId == valueId;
		SlotIdentity identity;
		return ObserveSlotIdentity(slot, &identity) &&
			identity.valueId == valueId;
	}

	uint64_t ReadSlotVersion(uint32_t slot) const
	{
		if (!versionPages_ || slot >= maxSlots_)
			return version_.load(std::memory_order_acquire);
		VersionPage *page = versionPages_[slot / VersionPageSize].load(
			std::memory_order_acquire);
		return page ?
			page->slots[slot & (VersionPageSize - 1)].version.load(
				std::memory_order_acquire) : 2;
	}

	uint64_t ReadVersion() const
	{
		return version_.load(std::memory_order_acquire);
	}

	// ---- Per-slot copy-gate bits ----
	// Four bits per slot, stored word-wise per 64-slot group. `stale` is the
	// only correctness bit: every reader path checks it, so `stale` clear
	// means the shadow is exact or absent. The other three only decide
	// whether the gate may skip a copy into the slot.
	enum GateBit : uint32_t { GateFollow = 1, GatePinned = 2, GateStale = 4 };

	// Missing page means no bits: stale clear, and no shadow was ever
	// mirrored there.
	uint32_t LoadGateBits(uint32_t slot) const noexcept
	{
		const GateGroup *group = PeekGroup(slot);
		if (!group)
			return 0;
		const uint64_t bit = SlotBit(slot);
		uint32_t bits = 0;
		if (group->follow.load(std::memory_order_relaxed) & bit)
			bits |= GateFollow;
		if ((group->pinCur.load(std::memory_order_relaxed) |
			group->pinPrev.load(std::memory_order_relaxed)) & bit)
			bits |= GatePinned;
		if (group->stale.load(std::memory_order_acquire) & bit)
			bits |= GateStale;
		return bits;
	}

	bool IsSlotStale(uint32_t slot) const noexcept
	{
		const GateGroup *group = PeekGroup(slot);
		return group &&
			(group->stale.load(std::memory_order_acquire) & SlotBit(slot));
	}

	// PinSlot/PinSlotRange/SetFollow/SetStale may throw bad_alloc: a gate
	// group is allocated with its version page, so page creation can fail.
	void SetFollow(uint32_t slot, bool value)
	{
		if (GateGroup *group = value ? EnsureGroup(slot) : PeekGroup(slot))
			StoreBit(group->follow, SlotBit(slot), value);
	}

	void SetStale(uint32_t slot, bool value)
	{
		if (GateGroup *group = value ? EnsureGroup(slot) : PeekGroup(slot))
			StoreBit(group->stale, SlotBit(slot), value);
	}

	void PinSlot(uint32_t slot) { PinSlotRange(slot, 1); }

	// Word-wise: one test and at most one OR per 64 slots.
	void PinSlotRange(uint32_t first, uint32_t count)
	{
		if (count == 0 || first >= maxSlots_)
			return;
		const uint32_t end = static_cast<uint32_t>(
			std::min<uint64_t>(uint64_t(first) + count, maxSlots_));
		for (uint32_t slot = first; slot < end; ) {
			const uint32_t groupEnd = std::min(end, (slot | 63u) + 1);
			const uint32_t n = groupEnd - slot;
			const uint64_t mask =
				(n == 64 ? ~0ull : ((1ull << n) - 1)) << (slot & 63u);
			if (GateGroup *group = EnsureGroup(slot))
				OrBits(group->pinCur, mask);
			slot = groupEnd;
		}
	}

	// Returns how many slots are pinned after rotation, and re-evaluates the
	// heap's gate switch with hysteresis (off above 50% of slots pinned, back
	// on below 35% so it does not flap).
	uint32_t RotatePinGenerations() noexcept
	{
		uint64_t pinned = 0;
		for (uint32_t i = 0; i < versionPageCount_; ++i) {
			VersionPage *page = versionPages_[i].load(std::memory_order_acquire);
			if (!page)
				continue;
			for (GateGroup &group : page->gate) {
				const uint64_t current =
					group.pinCur.exchange(0, std::memory_order_acq_rel);
				group.pinPrev.store(current, std::memory_order_release);
				pinned += std::popcount(current);
			}
		}
		const bool disabled = gatingDisabled_.load(std::memory_order_relaxed);
		if (!disabled && pinned * 2 > maxSlots_)
			gatingDisabled_.store(true, std::memory_order_relaxed);
		else if (disabled && pinned * 100 < uint64_t(maxSlots_) * 35)
			gatingDisabled_.store(false, std::memory_order_relaxed);
		return static_cast<uint32_t>(pinned);
	}

	bool GatingDisabled() const noexcept
	{
		return gatingDisabled_.load(std::memory_order_relaxed);
	}

	// Heap retirement or ID reuse: forget everything, including the hysteresis
	// switch, so a reused record starts clean.
	void ClearAllGatingBits() noexcept
	{
		for (uint32_t i = 0; i < versionPageCount_; ++i) {
			VersionPage *page = versionPages_[i].load(std::memory_order_acquire);
			if (!page)
				continue;
			for (GateGroup &group : page->gate) {
				group.follow.store(0, std::memory_order_relaxed);
				group.pinCur.store(0, std::memory_order_relaxed);
				group.pinPrev.store(0, std::memory_order_relaxed);
				group.stale.store(0, std::memory_order_release);
			}
		}
		gatingDisabled_.store(false, std::memory_order_relaxed);
	}

private:
	static constexpr uint32_t VersionPageSize = 256;
	static_assert((VersionPageSize & (VersionPageSize - 1)) == 0,
		"descriptor version page size must be a power of two");

	// One group covers 64 slots, so a whole-range pin or an epoch rotation is a
	// handful of word operations rather than a per-slot loop, and all four bits
	// of a group sit in one cache line. Declared at class scope, before
	// VersionPage, because the public gate API and PeekGroup/EnsureGroup name it.
	struct alignas(32) GateGroup
	{
		std::atomic<uint64_t> follow{0};
		std::atomic<uint64_t> pinCur{0};
		std::atomic<uint64_t> pinPrev{0};
		std::atomic<uint64_t> stale{0};
	};
	static_assert(VersionPageSize % 64 == 0);
	static_assert(sizeof(GateGroup) == 32);

	// Per-slot publication state is interleaved so that one cache line holds
	// the complete identity of two adjacent slots. The previous layout kept
	// four parallel arrays, so every publish/validate touched up to four
	// separate lines, and descriptor copy storms bounced all of them between
	// cores.
	struct alignas(CacheLineBytes) VersionPage
	{
		struct SlotState
		{
			std::atomic<uint64_t> version;
			std::atomic<uint64_t> valueId;
			// Flat post-publication snapshots, written after the seqlock
			// version store completes. Readers compare flatVersion against
			// the live version: when they match, the slot has not been
			// republished since the snapshot, so flatValueId is the
			// authoritative current valueId. This gives descriptor copy
			// tracking (RecordCopiedDescriptorsSimple)
			// a lock-free O(1) "is dest already identical to src" check
			// indexed by slot - the same design as DXVK's
			// views[slot] != resView compare - without the seqlock's
			// three-load read protocol or any hash-table capacity limits.
			std::atomic<uint64_t> flatValueId;
			std::atomic<uint64_t> flatVersion;
		};
		static_assert(sizeof(SlotState) * 2 == CacheLineBytes,
			"two slot states should fill one cache line");
		std::array<SlotState, VersionPageSize> slots;

		// Copy-gate bits live next to the version state so a page that exists
		// for slot publication also owns the gate words for its slots; pages
		// are allocated lazily, so a page with no writes carries no bits.
		std::array<GateGroup, VersionPageSize / 64> gate;

		VersionPage()
		{
			for (SlotState &slot : slots) {
				slot.version.store(2, std::memory_order_relaxed);
				slot.valueId.store(0, std::memory_order_relaxed);
				slot.flatValueId.store(0, std::memory_order_relaxed);
				slot.flatVersion.store(2, std::memory_order_relaxed);
			}
		}
	};

	VersionPage *EnsureVersionPage(uint32_t slot)
	{
		if (!versionPages_ || slot >= maxSlots_)
			return nullptr;
		std::atomic<VersionPage*> &cell =
			versionPages_[slot / VersionPageSize];
		VersionPage *page = cell.load(std::memory_order_acquire);
		if (page)
			return page;
		VersionPage *candidate = new VersionPage();
		VersionPage *expected = nullptr;
		if (cell.compare_exchange_strong(
				expected, candidate,
				std::memory_order_release,
				std::memory_order_acquire))
			return candidate;
		delete candidate;
		return expected;
	}

	static constexpr uint64_t SlotBit(uint32_t slot) noexcept
	{
		return 1ull << (slot & 63u);
	}

	// Index of the 64-slot group inside its page.
	static constexpr uint32_t GroupIndex(uint32_t slot) noexcept
	{
		return (slot & (VersionPageSize - 1)) >> 6;
	}

	const GateGroup *PeekGroup(uint32_t slot) const noexcept
	{
		if (!versionPages_ || slot >= maxSlots_)
			return nullptr;
		const VersionPage *page =
			versionPages_[slot / VersionPageSize].load(std::memory_order_acquire);
		return page ? &page->gate[GroupIndex(slot)] : nullptr;
	}

	GateGroup *PeekGroup(uint32_t slot) noexcept
	{
		return const_cast<GateGroup *>(std::as_const(*this).PeekGroup(slot));
	}

	// Throws bad_alloc like EnsureVersionPage does; callers that allocate
	// pages (SetFollow, SetStale, PinSlotRange) therefore cannot be noexcept.
	GateGroup *EnsureGroup(uint32_t slot)
	{
		VersionPage *page = EnsureVersionPage(slot);   // page index is slot / 256
		return page ? &page->gate[GroupIndex(slot)] : nullptr;
	}

	// Test before read-modify-write: a locked instruction is only paid when
	// the bit actually changes. Most copies leave it as it was.
	static void StoreBit(
		std::atomic<uint64_t> &word, uint64_t bit, bool value) noexcept
	{
		if (value) {
			if (!(word.load(std::memory_order_relaxed) & bit))
				word.fetch_or(bit, std::memory_order_release);
		} else if (word.load(std::memory_order_relaxed) & bit) {
			word.fetch_and(~bit, std::memory_order_release);
		}
	}

	static void OrBits(std::atomic<uint64_t> &word, uint64_t mask) noexcept
	{
		if ((word.load(std::memory_order_relaxed) & mask) != mask)
			word.fetch_or(mask, std::memory_order_relaxed);
	}

	std::atomic<bool> materialized_{false};
	std::atomic<bool> active_{true};
	// Gate switch of this heap, re-evaluated with the pin-generation rotation.
	std::atomic<bool> gatingDisabled_{false};
	std::array<LockCell, ShardCount> locks_;
	// Consumers only need a non-zero token that changes after any Descriptor
	// State write. Keeping 1024 per-shard counters made every cache validation
	// read 1024 cache lines; one heap-wide token makes that hot read O(1).
	alignas(CacheLineBytes) std::atomic<uint64_t> version_{1};
	uint32_t maxSlots_ = 0;
	uint32_t versionPageCount_ = 0;
	std::unique_ptr<std::atomic<VersionPage*>[]> versionPages_;
};


// Chunked dense current-value storage used by Descriptor State. This is the
// single source of truth for descriptor values in normal Mod mode.
//
// The previous design kept a sparse hash-like store (per-slot value nodes with
// refcounts) because whole-heap dense arrays would allocate megabytes for
// million-slot heaps. That store forced every descriptor write under the
// per-shard exclusive lock and paid a value-pool acquire/copy/release per slot.
//
// This store keeps the dense-array access pattern (one direct index, plain
// field copies, no per-slot allocation) but allocates the storage lazily in
// 4096-slot chunks: the heap owns only a small chunk-pointer table (2 KB for a
// million-slot heap) and each chunk body is CAS-allocated on first write to
// any of its slots. Memory therefore covers exactly the regions the game
// actually touches.
//
// The store deliberately contains no per-slot locking. The owning descriptor
// heap supplies that policy: writers serialize an individual slot through the
// shard lock while the chunk is missing, and are lock-free afterwards (D3D12
// requires applications to serialize writes to the same descriptor handle,
// exactly the single-writer assumption the materialized dense path already
// uses). Chunk publication is atomic because different slot locks can touch
// the same chunk for the first time concurrently.
template <typename Value>
class DX12ChunkedDescriptorStore
{
public:
	static constexpr uint32_t ChunkSlots = 4096;
	static_assert((ChunkSlots & (ChunkSlots - 1)) == 0,
		"descriptor chunk slot count must be a power of two");

	struct Chunk
	{
		std::array<Value, ChunkSlots> values = {};
		std::array<uint8_t, ChunkSlots> occupied = {};
	};

	DX12ChunkedDescriptorStore() = default;

	DX12ChunkedDescriptorStore(const DX12ChunkedDescriptorStore&) = delete;
	DX12ChunkedDescriptorStore &operator=(
		const DX12ChunkedDescriptorStore&) = delete;

	DX12ChunkedDescriptorStore(DX12ChunkedDescriptorStore &&other) noexcept
	{
		MoveFrom(std::move(other));
	}

	DX12ChunkedDescriptorStore &operator=(
		DX12ChunkedDescriptorStore &&other) noexcept
	{
		if (this == &other)
			return *this;
		Reset();
		MoveFrom(std::move(other));
		return *this;
	}

	~DX12ChunkedDescriptorStore()
	{
		Reset();
	}

	// Called once when the descriptor heap boundary becomes known. Only the
	// small chunk-pointer table is allocated here; chunk bodies stay lazy.
	void Initialize(uint32_t maxSlots)
	{
		Reset();
		maxSlots_ = maxSlots;
		chunkCount_ = maxSlots == 0 ? 0 : static_cast<uint32_t>(
			(static_cast<uint64_t>(maxSlots) + ChunkSlots - 1) / ChunkSlots);
		if (chunkCount_ == 0)
			return;
		chunks_ = std::make_unique<std::atomic<Chunk*>[]>(chunkCount_);
		for (uint32_t index = 0; index < chunkCount_; ++index)
			chunks_[index].store(nullptr, std::memory_order_relaxed);
	}

	// CAS-allocates the chunk containing slot on first touch. Callers already
	// serialize the owning slot, so the returned chunk address remains stable.
	Chunk *EnsureChunkForSlot(uint32_t slot)
	{
		if (!chunks_ || slot >= maxSlots_)
			return nullptr;
		std::atomic<Chunk*> &cell = chunks_[slot / ChunkSlots];
		Chunk *chunk = cell.load(std::memory_order_acquire);
		if (chunk)
			return chunk;

		Chunk *candidate = new Chunk();
		Chunk *expected = nullptr;
		if (cell.compare_exchange_strong(
				expected, candidate,
				std::memory_order_release,
				std::memory_order_acquire))
			return candidate;
		delete candidate;
		return expected;
	}

	Chunk *ChunkForSlot(uint32_t slot)
	{
		return const_cast<Chunk*>(
			static_cast<const DX12ChunkedDescriptorStore*>(this)->
				ChunkForSlot(slot));
	}

	const Chunk *ChunkForSlot(uint32_t slot) const
	{
		if (!chunks_ || slot >= maxSlots_)
			return nullptr;
		return chunks_[slot / ChunkSlots].load(std::memory_order_acquire);
	}

	bool ChunkExists(uint32_t slot) const
	{
		return ChunkForSlot(slot) != nullptr;
	}

	// Raw slot pointer when the chunk exists, regardless of occupancy.
	// Lock-free writers/readers use this under the per-slot seqlock contract.
	Value *SlotPtr(uint32_t slot)
	{
		return const_cast<Value*>(
			static_cast<const DX12ChunkedDescriptorStore*>(this)->
				SlotPtr(slot));
	}

	const Value *SlotPtr(uint32_t slot) const
	{
		const Chunk *chunk = ChunkForSlot(slot);
		return chunk ? &chunk->values[slot & (ChunkSlots - 1)] : nullptr;
	}

	// Occupied-slot access.
	Value *Find(uint32_t slot)
	{
		return const_cast<Value*>(
			static_cast<const DX12ChunkedDescriptorStore*>(this)->Find(slot));
	}

	const Value *Find(uint32_t slot) const
	{
		const Chunk *chunk = ChunkForSlot(slot);
		if (!chunk)
			return nullptr;
		const uint32_t offset = slot & (ChunkSlots - 1);
		return chunk->occupied[offset] ? &chunk->values[offset] : nullptr;
	}

	bool IsOccupied(uint32_t slot) const
	{
		return Find(slot) != nullptr;
	}

	void SetOccupied(uint32_t slot, bool occupied)
	{
		Chunk *chunk = ChunkForSlot(slot);
		if (chunk)
			chunk->occupied[slot & (ChunkSlots - 1)] = occupied ? 1 : 0;
	}

	template <typename Visitor>
	void ForEachOccupied(Visitor visitor)
	{
		for (uint32_t chunkIndex = 0; chunkIndex < chunkCount_; ++chunkIndex) {
			Chunk *chunk = chunks_[chunkIndex].load(std::memory_order_acquire);
			if (!chunk)
				continue;
			const uint32_t firstSlot = chunkIndex * ChunkSlots;
			const uint32_t slotsInChunk = firstSlot + ChunkSlots <= maxSlots_ ?
				ChunkSlots : maxSlots_ - firstSlot;
			for (uint32_t offset = 0; offset < slotsInChunk; ++offset) {
				if (chunk->occupied[offset])
					visitor(firstSlot + offset, chunk->values[offset]);
			}
		}
	}

	// Releases every chunk and the pointer table. Only called when the owning
	// heap record is retired or tracking is cleaned up; chunk bodies are never
	// released at materialization so in-flight lock-free accesses cannot dangle.
	void Reset()
	{
		if (!chunks_)
			return;
		for (uint32_t index = 0; index < chunkCount_; ++index) {
			Chunk *chunk = chunks_[index].exchange(
				nullptr, std::memory_order_acq_rel);
			delete chunk;
		}
		chunks_.reset();
		chunkCount_ = 0;
		maxSlots_ = 0;
	}

private:
	void MoveFrom(DX12ChunkedDescriptorStore &&other)
	{
		chunks_ = std::move(other.chunks_);
		chunkCount_ = other.chunkCount_;
		maxSlots_ = other.maxSlots_;
		other.chunkCount_ = 0;
		other.maxSlots_ = 0;
	}

	std::unique_ptr<std::atomic<Chunk*>[]> chunks_;
	uint32_t chunkCount_ = 0;
	uint32_t maxSlots_ = 0;
};


