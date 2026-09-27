# Optimizer Data Pipeline Analogy

The data pipeline architecture for the Optimizer maps perfectly to reading books in a library:

*   **The Library = The Hard Drive (Disk).** It holds all the data, but it's too big to read all at once.
*   **The Book = The RAM Buffer.** You pull one 100,000-token chunk from the library and put it on your desk.
*   **The Page = The Context Window.** You look at exactly 128 tokens (512 elements) at a time, hand it to the AI, and then shift to the next page.
*   **Ripping the last pages out = The Carry-Forward.** Before you throw the finished book away to grab the next one from the library, you rip out the last 128 tokens and staple them to the front of the *new* book. This guarantees the AI's "story" is never interrupted when crossing the boundary between books!
