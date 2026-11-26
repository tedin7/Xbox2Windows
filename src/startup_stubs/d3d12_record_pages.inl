// Memory pages for replay recordings (docs/startup-runtime.md#d3d12-replay-record): the hooks the recorder calls,
// filled by the page store (PAGE/MAP chunks). Each hook appends its chunks to the caller's thread buffer `b`, whose lock
// the caller holds, under the sequence number `seq` the replay applies them at (before the call or mark with that seq).
// Included inside namespace replay_record by d3d12_record.inl.

// The window has just opened at present `present`: the base image of the memory a replay needs.
inline void pages_at_window_open(ThreadBuffer& b, u64 seq, LONG present)
{
    (void)b, (void)seq, (void)present;
}

// A present inside the window: the pages changed since the last capture.
inline void pages_at_present(ThreadBuffer& b, u64 seq, LONG present)
{
    (void)b, (void)seq, (void)present;
}
