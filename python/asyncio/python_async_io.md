# Python `asyncio` From Zero

## Event loops, coroutines, `await`, Tasks, `gather()`, and `to_thread()`—with real backend examples

You open a Python backend and see this:

```python
async def process_request():
    result = await some_api_call()
```

What is really happening? This guide builds the mental model from scratch, then applies it to realistic API and AI-backend code.

```text
normal Python
    ↓
async Python
    ↓
coroutines + event loop
    ↓
await + Tasks
    ↓
concurrent I/O
    ↓
sync-library bridge: to_thread()
```

> Short version: `asyncio` is excellent when an application spends a lot of time waiting for networks, databases, caches, files, or other services. While one operation waits, the event loop can run another ready operation.

---

## 1. The problem: waiting wastes a server thread

Consider a normal synchronous database request:

```python
def get_user(user_id):
    return database.get_user(user_id)

def main():
    user = get_user(42)
    print(user)
```

Flow:

```text
main()
  ↓
get_user()
  ↓
database.get_user()
  ↓
WAIT FOR DATABASE
  ↓
database responds
  ↓
print(user)
```

During the database wait, this thread is blocked. For one small script, that is fine. For a server handling many requests, many operations may be waiting on external systems:

```text
PostgreSQL     Redis     HTTP service     object storage     LLM API
    │            │             │                │             │
  waiting      waiting       waiting          waiting       waiting
```

Async programming says: **while one operation is waiting, let the application make progress on another operation.**

---

## 2. The event loop

`asyncio` uses an **event loop**: a scheduler that runs asynchronous work and revisits it when the thing it was waiting for becomes ready.

```text
                       EVENT LOOP
                            │
             ┌──────────────┼──────────────┐
             ↓              ↓              ↓
          Task A         Task B         Task C
         DB request      LLM call       Redis read
             │              │              │
          waiting        waiting        waiting
```

Usually, one event loop runs in one OS thread. It runs one piece of Python code at a time in that thread. When the current task reaches a real suspension point, the loop may run another **ready** task.

That last word matters: if nothing else is ready, the loop simply waits. Async does not make one request inherently faster; it makes waiting time shareable.

---

## 3. `async def`: a coroutine function

```python
async def hello():
    print("Hello")
```

This defines a **coroutine function**. Calling it does not normally run its body immediately:

```python
coro = hello()
```

`coro` is a **coroutine object**—a description of work that has not been driven by the event loop yet.

```text
async def hello()  → coroutine function
hello()            → coroutine object
await hello()      → run it as part of the current coroutine
```

This common mistake creates an un-awaited coroutine:

```python
async def main():
    hello()            # creates work, but does not run it

asyncio.run(main())
```

Python typically emits `RuntimeWarning: coroutine 'hello' was never awaited`.

---

## 4. `asyncio.run()`: the program entry point

For a standalone program, start one top-level coroutine like this:

```python
import asyncio

async def main():
    print("starting")

asyncio.run(main())
```

Conceptually:

```text
asyncio.run(main())
        ↓
creates and manages an event loop
        ↓
runs main() to completion
        ↓
cleans up the loop
```

Do not use `asyncio.run()` inside a running event loop (for example, in many notebooks or async web frameworks). There, the framework already owns the loop; write `async def` code and let the framework call it.

---

## 5. `await`: pause this coroutine until a result is ready

```python
import asyncio

async def slp():
    await asyncio.sleep(5)
    print("k")

async def main():
    print("i")
    await slp()
    print("j")

asyncio.run(main())
```

Output:

```text
i
# five seconds later
k
j
```

Flow:

```text
main()
  ↓
print("i")
  ↓
await slp()
  ↓
slp() reaches await asyncio.sleep(5)
  ↓
slp() suspends; therefore main() is waiting too
  ↓
event loop can run other ready tasks
  ↓
sleep completes → slp() resumes → prints k → returns
  ↓
main() resumes → prints j
```

`await` means: “I need the result of this awaitable before continuing this coroutine.”

### Important nuance: `await` does not always switch tasks

An `await` only gives the event loop an opportunity to run something else if the awaited operation actually suspends. This runs straight through:

```python
async def hello():
    print("hello")

async def main():
    await hello()
    print("done")
```

Output is simply `hello`, then `done`. There is no waiting operation inside `hello()`.

---

## 6. A coroutine is not a thread

Creating many coroutines does **not** create many OS threads.

```text
One OS thread
     │
  Event loop
     ├── coroutine A
     ├── coroutine B
     ├── coroutine C
     └── coroutine D
```

Each coroutine is sequential within itself. It runs until it returns or reaches an awaitable that suspends. This is **cooperative scheduling**: code must cooperate by using non-blocking async libraries and by reaching suspension points in a reasonable time.

---

## 7. Direct `await` versus `create_task()`

### Direct `await`: run it and wait immediately

```python
result = await fetch_profile(user_id)
```

Your coroutine will not move past that line until `fetch_profile()` finishes.

### `asyncio.create_task()`: schedule independent work

```python
profile_task = asyncio.create_task(fetch_profile(user_id))

# Continue doing work that does not need the profile yet.
audit = build_audit_record()

profile = await profile_task
```

`create_task()` wraps a coroutine in a **Task** and schedules it on the current event loop.

```text
fetch_profile()                 → coroutine object
create_task(fetch_profile())    → scheduled Task
await task                      → wait for its result
```

### A small output example

```python
import asyncio

async def process_file(path):
    data = await asyncio.to_thread(read_large_file, path)
    print("i")
    return process(data)

async def main():
    print("j")
    task = asyncio.create_task(process_file("report.csv"))
    print("k")
    await task
```

The normal ordering is:

```text
j
k
i
```

`create_task()` schedules `process_file`, but `main()` continues to `print("k")` before it waits for the task. Exact ordering among independently scheduled tasks should not be used as application logic.

---

## 8. `asyncio.gather()`: wait for several independent results

Suppose an API endpoint needs an LLM response and user preferences. Neither depends on the other.

Sequential version:

```python
llm = await call_llm(message)          # 500 ms
preferences = await get_preferences()  # 300 ms
```

```text
LLM:          ────────── 500 ms
Preferences:              ────── 300 ms
Total:                     ~800 ms
```

Concurrent version:

```python
llm, preferences = await asyncio.gather(
    call_llm(message),
    get_preferences(user_id),
)
```

```text
LLM:          ────────── 500 ms
Preferences:   ────── 300 ms
Total:         ~500 ms
```

`gather()` is ideal when you need all results and the operations can begin independently.

```text
                   Event loop
                       │
              ┌────────┴────────┐
              ↓                 ↓
         call_llm()       get_preferences()
              │                 │
           waiting           waiting
              └────────┬────────┘
                       ↓
                gather returns both
```

By default, an exception from one awaited operation is propagated by `gather()`. For batch-style work where you want every result, use `return_exceptions=True` deliberately and inspect the results; do not silently treat exceptions as valid values.

---

## 9. Realistic AI-backend flow

An AI assistant may need user data, history, and preferences before calling the model. These reads can happen concurrently; generating a reply depends on them.

```python
import asyncio

async def process_request(user_id: str, message: str) -> str:
    user_task = asyncio.create_task(db.get_user(user_id))
    history_task = asyncio.create_task(db.get_history(user_id))
    preferences_task = asyncio.create_task(redis.get(f"preferences:{user_id}"))

    user, history, preferences = await asyncio.gather(
        user_task,
        history_task,
        preferences_task,
    )

    response = await llm.generate(
        message,
        user=user,
        history=history,
        preferences=preferences,
    )

    await db.save_response(user_id, response)
    return response
```

```text
User request
    ↓
start user / history / preferences reads together
    ↓
wait until all are ready
    ↓
call LLM with gathered context
    ↓
save response
    ↓
return response
```

Nothing is “automatically async” because it touches a database or network. `db.get_user`, `redis.get`, and `llm.generate` must come from libraries that provide genuine async, non-blocking operations.

---

## 10. Why Python cannot automatically make every slow function async

Given this call:

```python
result = mysterious_function()
```

Python cannot reliably know whether it is doing network I/O, disk I/O, a quick calculation, or a ten-second CPU loop:

```python
def mysterious_function():
    time.sleep(10)          # blocking wait

def another_function():
    for _ in range(10**10): # CPU-heavy work
        calculate()
```

Arbitrary synchronous code is allowed to run until it returns. The interpreter cannot safely pause it at random lines, switch to another coroutine, and later resume it. Async libraries are designed to expose safe suspension points; that is why async is cooperative.

---

## 11. The bridge for blocking code: `asyncio.to_thread()`

Sometimes an async application must use a synchronous library:

```python
def old_api_call():
    response = requests.get("https://example.com")
    return response.json()
```

Calling it directly from an async handler blocks the event-loop thread:

```python
async def handler():
    data = old_api_call()  # blocks the event loop: avoid
    return data
```

Instead:

```python
async def handler():
    data = await asyncio.to_thread(old_api_call)
    return data
```

`to_thread()` runs a normal callable in a worker thread and gives you an awaitable result.

```text
Event-loop thread                      Worker thread
-----------------                      -------------
await to_thread(old_api_call)  ───→    old_api_call()
       │                                      ↓
       │                                 blocking network wait
       ↓                                      ↓
run other ready tasks             ←── result / exception
       ↓
resume this coroutine
```

### Why not `await old_api_call()`?

This is invalid for a normal function and would block before `await` could help:

```python
def old_api_call():
    time.sleep(5)
    return "done"

# The function call must happen first, so it blocks the event loop.
await old_api_call()
```

Use `await asyncio.to_thread(old_api_call)` instead.

### Passing arguments

`to_thread()` accepts positional and keyword arguments for the callable:

```python
document = await asyncio.to_thread(
    s3_client.get_object,
    Bucket="documents",
    Key=document_id,
)
```

---

## 12. Real `to_thread()` use cases

### Legacy SDK in an async API

```python
def search_documents(query: str):
    return legacy_search_client.search(query)

async def agent(message: str):
    answer = await llm.generate(message)
    documents = await asyncio.to_thread(search_documents, answer)
    return documents
```

### Synchronous cloud client

```python
async def get_document(document_id: str):
    return await asyncio.to_thread(
        s3_client.get_object,
        Bucket="documents",
        Key=document_id,
    )
```

### Slow filesystem operation

```python
def read_large_file(path: str) -> str:
    with open(path, encoding="utf-8") as file:
        return file.read()

async def process_file(path: str):
    data = await asyncio.to_thread(read_large_file, path)
    return parse_document(data)
```

### Sync database driver during a migration

You can put a bounded blocking DB call in a thread while modernizing an application, but a production service is often better served by a proper async driver and carefully managed connection pool. Do not share non-thread-safe cursors or client objects without checking the library’s rules.

### What `to_thread()` is not for

It is mostly for **I/O-bound blocking functions**. A pure-Python CPU-heavy loop still contends with the GIL, so threads usually will not provide true CPU parallelism:

```python
def huge_calculation():
    for _ in range(10**10):
        calculate()
```

For serious CPU-bound work, consider process-based workers, `ProcessPoolExecutor`, a job queue, or a library that releases the GIL.

---

## 13. Concurrency versus parallelism

These words are related but different.

| Term | Meaning | Typical asyncio example |
| --- | --- | --- |
| Concurrency | Multiple operations make progress over overlapping time. | One task waits for a database while the event loop serves another task. |
| Parallelism | Multiple computations execute literally at the same time on multiple CPU cores. | Separate processes working on CPU-heavy jobs. |

```text
Concurrency (one event-loop thread)
Task A: run → wait ─────────────→ resume
Task B:       run → wait ─→ resume

Parallelism (multiple CPU workers)
CPU 1: calculation A ███████████
CPU 2: calculation B ███████████
```

`asyncio` provides concurrency. It is particularly effective when time is dominated by waiting rather than CPU computation.

---

## 14. Sequential does not mean synchronous

This coroutine runs in source-code order:

```python
async def main():
    print("A")
    await something()
    print("B")
```

But it is asynchronous because it can suspend at the `await`, letting the event loop run other ready tasks.

```text
Synchronous function
  sequential + blocks its thread while waiting

Async coroutine
  sequential + can suspend without blocking the event-loop thread
```

If `main()` awaits `process_file()`, `main()` waits for it. If `process_file()` is currently awaiting `to_thread(...)`, both are paused, but the event loop is free to run other tasks.

---

## 15. Common mistakes

### Mistake: forgetting `await`

```python
async def main():
    fetch_profile()  # not executed
```

Fix:

```python
profile = await fetch_profile()
```

Or explicitly schedule it:

```python
task = asyncio.create_task(fetch_profile())
```

### Mistake: using `await` inside `def`

```python
def main():
    await slp()  # SyntaxError
```

Fix:

```python
async def main():
    await slp()
```

### Mistake: calling blocking code directly in an async function

```python
async def endpoint():
    return requests.get("https://example.com")  # blocks loop
```

Fix: use an async HTTP client, or as a transition use `await asyncio.to_thread(requests.get, url)`.

### Mistake: creating background tasks and losing them

```python
asyncio.create_task(send_analytics())
```

If the task matters, keep a reference and await or otherwise supervise it. Unobserved task exceptions are easy to miss, and the event loop may shut down before fire-and-forget work completes.

### Mistake: assuming `create_task()` means a new thread

It does not. It schedules a coroutine on the same event loop.

### Mistake: blocking the loop with CPU work

```python
async def endpoint():
    return expensive_pure_python_calculation()  # blocks loop
```

Move CPU-heavy work to a process or specialized worker system.

### Mistake: adding concurrency to dependent steps

Do not start work concurrently if step B needs the result of step A:

```python
user = await get_user(user_id)
recommendations = await get_recommendations(user)  # correctly dependent
```

---

## 16. A practical decision guide

```text
Is the operation already async and non-blocking?
  └─ Yes → await it

Do several independent async operations need to finish?
  └─ Yes → gather(...) or create_task(...) + await

Is it a synchronous, potentially blocking I/O call?
  └─ Yes → prefer an async library; otherwise use to_thread(...) carefully

Is it CPU-heavy work?
  └─ Yes → use processes / a worker queue / native code that releases the GIL
```

---

## 17. Final cheat sheet

```python
async def work():
    ...
```

Defines a coroutine function.

```python
coro = work()
```

Creates a coroutine object; it has not necessarily run.

```python
result = await work()
```

Runs it as part of the current coroutine and waits for its result.

```python
task = asyncio.create_task(work())
```

Schedules it independently on the event loop.

```python
result = await task
```

Waits for a Task’s result.

```python
a, b = await asyncio.gather(work_a(), work_b())
```

Starts independent async operations together and waits for both.

```python
result = await asyncio.to_thread(sync_function, arg)
```

Runs blocking synchronous work in a worker thread so it does not block the event-loop thread.

```text
asyncio: efficient concurrency for waiting-heavy work
create_task/gather: overlap independent async operations
to_thread: bridge blocking synchronous I/O into an async application
processes: better default for CPU-heavy Python work
```

The one sentence worth remembering:

> **`asyncio` lets one thread efficiently wait for many things; `to_thread()` lets that async application safely wait for blocking code that does not cooperate with the event loop.**
