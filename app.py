"""
Root-level app.py — thin re-export for backward compatibility.

The actual server lives at district_accord/server/app.py.
This file lets `uvicorn app:app` and the Dockerfile CMD keep working.
"""

from district_accord.server.app import app, main  # noqa: F401

if __name__ == "__main__":
    main()
