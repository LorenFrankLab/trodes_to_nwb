# Ruff Issues Fix Plan

This document tracks the plan to fix the remaining 56 ruff issues (excluding notebook issues).

## 🔴 Priority 1: Critical Fixes (7 issues) - ✅ COMPLETED

### Immediate Action Required
- [x] **Mutable Default Argument** (`convert_ephys.py:42`)
  - Change `nwb_hw_channel_order=[]` to `nwb_hw_channel_order=None`
  - Add `if nwb_hw_channel_order is None: nwb_hw_channel_order = []` inside function

- [x] **Missing Raise Statements** (2 issues)
  - `spike_gadgets_raw_io.py:170, 1210` - Add `raise` keyword before exception instantiation

- [x] **Exception Chaining** (`convert_position.py:134, 602`)
  - Change `raise SomeException(...)` to `raise SomeException(...) from err`

- [x] **Top-Level Imports** (`convert_optogenetics.py` - 4 locations)
  - Move `import` statements from inside functions to module top

## 🟡 Priority 2: Code Quality (25 issues) - ✅ COMPLETED

### Quick Wins - Auto-fixable patterns
- [x] **Dictionary/List Inefficiencies** (11 issues)
  - Replace `key in dict.keys()` with `key in dict` (8 instances)
  - Replace `dict()` with `{}` literals (2 instances)
  - Replace list comprehension with set comprehension (1 instance)

- [x] **Logic Simplification** (6 issues)
  - Use ternary operators for simple if/else blocks
  - Use `.get()` method instead of if/else for dict access
  - Replace `not a == b` with `a != b`

- [x] **Unused Variables** (6 issues)
  - Remove unused assignments in tests
  - Replace unused loop variables with `_`

- [x] **Unnecessary Comprehensions** (6 issues)
  - Convert list comprehensions to generators where appropriate

## 🟠 Priority 3: Style & Performance (24 issues) - PENDING

### Consider for future refactoring
- [ ] **Magic Numbers** (`convert_position.py` - 4 instances)
  - Extract constants: `MIN_TIMESTAMPS = 2`, `DEFAULT_TIMEOUT = 2000`, `MIN_TICKS = 100`

- [ ] **Memory Optimization** (`spike_gadgets_raw_io.py` - 4 instances)
  - Replace `@lru_cache` with `@cached_property` or manual caching for methods

- [ ] **Variable Naming** (2 instances)
  - Rename single-letter variables to descriptive names

- [ ] **Other Improvements** (6 issues)
  - Add stacklevel to warnings
  - Use contextlib.suppress() for clean exception handling
  - Remove unused imports

## Progress Tracking

**Total Issues**: 56 (excluding notebooks)
- **Fixed**: 44 (7 Priority 1 + 37 Priority 2)
- **Remaining**: 12

**Estimated Timeline**:
- Phase 1 (Critical): 30 minutes
- Phase 2 (Quality): 45 minutes
- Phase 3 (Style): As needed during regular development

## Commit Strategy

Each priority level will be committed separately with detailed commit messages explaining the fixes applied.