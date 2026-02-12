# color-correction

This tool **automatically color-corrects and enhances image files** in a folder.
Everything below is meant to be **copy → paste → run** friendly (no admin needed).

> **Command names:** This repo installs:
>
> * **`color-correction`** (default)
> * **`color-correction-a`** (variation A)
> * **`color-correction-b`** (variation B)

---

## What It Does

When you run any of the commands, it:

* Looks for a folder named **`input`** (unless you override `--input`)
* Recursively scans for image files
* Supports **JPEG / PNG** and **RAW formats** (ARW, DNG, CR2, CR3, NEF)
* Automatically:

  * Applies white balance
  * Corrects exposure
  * Lifts midtones
  * Protects highlights
  * Adjusts contrast and vibrance
* Writes corrected images as **JPEGs** to an **`output/`** folder (unless you override `--output`)
* Prints progress and stats in Terminal as it runs

---

## What You Need

* macOS
* **Python 3.9+**

You do **not** need:

* Admin/root access
* Coding experience
* Any extra apps

---

## Installation

### 1) Open Terminal

Press:

**⌘ + Space → type “Terminal” → press Enter**

---

### 2) Install the tool (copy + paste)

```bash
python3 -m pip install --user --upgrade pip
python3 -m pip install --no-cache-dir "git+https://github.com/chickensavory/color-correction.git"
```

---

### 3) Verify it installed

```bash
color-correction --help
color-correction-a --help
color-correction-b --help
```

If you see `command not found`, do the one-time setup below.

---

### 4) One-time setup (only if command is not found)

Find where Python installs user commands:

```bash
python3 -m site --user-base
```

If it prints something like:

```
/Users/YOURNAME/Library/Python/3.9
```

Run:

```bash
echo 'export PATH="$HOME/Library/Python/3.9/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
hash -r
```

Now try again:

```bash
color-correction --help
```

---

## How To Use (Step-by-Step)

### Step 1 — Create a working folder

```bash
mkdir color_work
cd color_work
```

---

### Step 2 — Create an `input` folder

```bash
mkdir input
```

---

### Step 3 — Add your images

Using Finder:

1. Open the `color_work` folder
2. Drag your images (or folders of images) into `input`

The tool will scan **all subfolders automatically**.

---

### Step 4 — Run the tool

Default pipeline:

```bash
color-correction
```

Variation A:

```bash
color-correction-a
```

Variation B:

```bash
color-correction-b
```

---

### Step 5 — Done

* Corrected images are written to **`output/`**
* Folder structure is preserved
* All outputs are saved as **JPEGs**
* Progress and stats are printed in Terminal

---

## Optional Arguments

You can override defaults if needed:

```bash
color-correction \
  --input input \
  --output output \
  --quality 95 \
  --contrast 1.1 \
  --vibrance 0.2
```

Works the same for variations:

```bash
color-correction-a --input input --output output_a
color-correction-b --input input --output output_b
```

Common options:

* `--input` → Input folder
* `--output` → Output folder
* `--quality` → JPEG quality (1–100)
* `--contrast` → Contrast strength
* `--vibrance` → Color vibrance
* `--sharpness` → Unsharp-mask amount
* `--pop` → Subject pop strength
* `--microcontrast` → Subject microcontrast amount
* `--micro-radius` → Microcontrast blur radius
* `--shadows` → Shadows lift (0..1)
* `--shadows-end` → Shadows fade-out luma
* `--raw-auto-wb` → Use rawpy auto WB (instead of camera WB)
* `--opencv-wb` → Enable OpenCV white balance
* `--opencv-wb-mode` → `simple` or `grayworld`
* `--no-bg-whiten` → Disable background whitening failsafe
* `--bg-whiten-target` → Background whitening target
* `--bg-whiten-max` → Max whitening gain
* `--xrite-hint` → Filename hint for the xrite anchor image
* `--no-skip-xrite-export` → Export xrite anchor output too
* `--xrite-required` → Fail if xrite anchor is missing
* `--wb-strength` → Anchor WB strength (0..1)
* `--wb-sat-max` → Max saturation for neutral-pixel WB sampling
* `--wb-y-min` → Min luma for neutral-pixel WB sampling
* `--wb-y-max` → Max luma for neutral-pixel WB sampling
* `--wb-min-pixels` → Minimum neutral pixels required

Run this to see all options:

```bash
color-correction --help
```

---

## Updating

```bash
python3 -m pip install --user --upgrade --no-cache-dir "git+https://github.com/chickensavory/color-correction.git"
```

---

## Troubleshooting

### Problem: `pip` not found

```bash
python3 -m ensurepip --upgrade
```

Then reinstall.

---

### Problem: `command not found: color-correction`

Follow the **One-time setup** step in Installation to add Python’s user `bin` folder to your PATH.

---

### Problem: RAW files not processed

Ensure `rawpy` installed correctly:

```bash
python3 -m pip install --user rawpy
```

If RAW is unavailable, the tool will warn and skip RAW files.

---

## Notes

* The tool expects an **`input/`** folder in the current directory unless `--input` is provided
* Output is always written as **JPEG**
* Originals are never modified
* First run on new images? Keep a backup, just in case
* Designed for fast, consistent product-style image correction