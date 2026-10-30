# Pure-Python single process with in-process inference

The obvious path by sibling-repo precedent was a Go daemon calling a Python
classifier over HTTP (auto-github's shape). We decided against it: the ML half
is unavoidably Python, the dataset/training/eval tooling is Python, and a Go
daemon would reintroduce a serialization boundary plus a
Go-formats-input-vs-Python-tokenizes drift risk (auto-github needs a whole
gotcha to police exactly this). One `gmail-labeler` Python package loads the
model in-process; train and inference share the single preprocessing function
by construction. Accepted cost: a resident ~1–2 GB RAM daemon and a
torch-bearing dependency stack for installers.
