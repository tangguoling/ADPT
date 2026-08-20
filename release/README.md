# Protected release build

The private development tree remains ordinary Python. A release build compiles
the GUI runtime, training backend, and prediction backend into native CPython
3.12 extensions, then removes those source files only from the staged bundle.

The project `model/` directory is excluded by default. Publish approved model
weights as a separate, versioned, signed asset so the same model can accompany
the matching Linux and Windows runtime bundles.

## Build prerequisites

Linux/WSL:

```bash
sudo apt install -y build-essential python3.12-dev
python -m pip install -r release/requirements-build.txt
```

Windows requires the **Microsoft C++ Build Tools** with the current MSVC C++
toolset and Windows SDK, followed by:

```powershell
python -m pip install -r release/requirements-build.txt
```

## Research release build

The Research Edition includes the non-commercial research-use license from
`RESEARCH_LICENSE.txt`:

```bash
python release/build_protected_release.py \
  --edition research
```

Commercial packaging remains an internal engineering path until its separate
terms have been finalized. An internal-only test build can be produced with:

```powershell
python release/build_protected_release.py `
  --edition commercial `
  --accept-draft-commercial-license
```

Run the command independently on Linux/WSL and Windows. Output is written to
`dist-protected/` and is specific to CPython 3.12 and the build platform.

Before distributing a Research Edition build:

1. confirm every protected source file is owned by or relicensable by the ADPT
   copyright holders;
2. review third-party notices against the exact release environment;
3. test the bundle in a clean Python 3.12 environment;
4. publish its SHA-256 digest; and
5. retain the exact private source revision used for the build.
