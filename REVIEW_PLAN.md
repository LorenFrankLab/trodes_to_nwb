# GitHub Review Response Plan

This document tracks the plan to address feedback from the GitHub PR review.

## 🔴 **Critical Security Issues** - ✅ COMPLETED

### 1. **Subprocess Security Vulnerabilities** (2 instances)
**Location**: `convert_position.py:1155, 1199`
**Issue**: Using `shell=True` with user-controlled input creates command injection risk
**Fix**: Replace with list-based subprocess calls
```python
# Before:
subprocess.run(f"ffmpeg -i {file} {new_file_name}", check=False, shell=True)
subprocess.run(f"cp {file} {new_file_name}", check=False, shell=True)

# After:
subprocess.run(['ffmpeg', '-i', file, new_file_name], check=False)
subprocess.run(['cp', file, new_file_name], check=False)
```

## 🟡 **Breaking Changes** - ✅ COMPLETED

### 2. **Channel Order Parameter Change** (convert_ephys.py:152)
**Issue**: Made previously optional parameter mandatory, breaking backward compatibility
**Action**: Revert to original default behavior with `np.arange(self.n_channel)`
**Maintainer feedback**: "Agree maybe revert back"

### 3. **Exception Type Change** (convert_position.py:193)
**Issue**: Changed `IOError` to `OSError` - need to verify compatibility
**Action**: Research if this change is safe or should be reverted
**Maintainer feedback**: "Need to verify this change"

## 🟠 **Code Quality Issues** - ✅ COMPLETED

### 4. **Import Sorting Issue** (convert_ephys.py:5)
**Issue**: New diagnostic shows imports are incorrectly sorted
**Fix**: Run `ruff check --fix` on the specific file

### 5. **Zip Safety Parameters** (test_convert_yaml.py:301, 389)
**Issue**: Added `strict=False` weakens safety guarantees
**Action**: ✅ Removed `strict=False` to use default `strict=True` behavior for better safety

### 6. **Unused Enumerate Variables** (convert_yaml.py:198)
**Issue**: Changed to `_probe_counter` but could remove enumerate entirely
**Action**: Remove `enumerate()` if index not needed

## 🔧 **Implementation Order**

1. **Fix security vulnerabilities** (subprocess calls)
2. **Revert breaking changes** (channel order, possibly exception type)
3. **Address code quality issues** (imports, zip, enumerate)
4. **Test all changes** to ensure no regressions
5. **Update PR** with fixes

## 📝 **Notes**

- Maintainer acknowledged security fixes as "reasonable" and "same"
- Breaking changes need careful consideration for backward compatibility
- Code quality fixes are less critical but improve maintainability