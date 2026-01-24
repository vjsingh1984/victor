# Quality Assurance Report

**Generated:** 2026-01-24T11:41:24.912791
**Version:** 0.5.0

## Summary

- **Total Checks:** 11
- **Passed:** 6
- **Failed:** 5
- **Success Rate:** 54.5%

## Detailed Results

### ✗ FAIL Integration Tests Execution

- **Duration:** 285.75s
- **Status:** Failed
- **Errors:**
  - Integration tests failed
  - victor/workflows/validation/tool_validator.py                           94     94     30      0   0.00%   48-364
  - victor/workflows/validation_rules.py                                   305    222    178      0  17.18%   83, 153, 187-220, 246-266, 282-330, 346-397, 431-519, 543-602, 624-652, 672-719, 735-762, 784-836, 841-871
  - victor/workflows/validator.py                                           45     23     10      0  40.00%   77-89, 103-109, 120, 131, 146, 154, 162, 170, 178, 187, 195, 203
  - victor/workflows/versioning.py                                         262    164     88      0  28.00%   118-123, 139-154, 173, 179, 183, 187, 240-256, 284, 295-298, 306-317, 348-349, 353-359, 364, 368, 410-412, 420-429, 441-446, 462-469, 480-494, 505-509, 521-522, 535-541, 559-588, 608-624, 642-648, 661-666, 685-693, 707-747
  - victor/workflows/visualization.py                                      618    540    270      0   8.78%   93, 99-109, 237-240, 244-354, 365-448, 459-505, 513-579, 590-687, 695-781, 799-822, 826-854, 867-898, 902-957, 961-981, 997-1014, 1018-1037, 1041-1063, 1067-1080, 1098-1123, 1141-1142, 1151, 1164-1170, 1175-1180
  - victor/workflows/yaml_loader.py                                        689    343    288     60  42.58%   234-235, 319, 437, 439, 441, 443, 445, 447, 449, 451, 452->454, 455, 457, 459, 487-564, 637-646, 696-735, 757-767, 847->898, 853-856, 880, 884, 888, 900, 909, 924-944, 958->956, 959, 964-968, 971-983, 988, 991-994, 999, 1002->1008, 1005, 1009, 1011-1013, 1017->1019, 1020-1038, 1043-1047, 1050-1052, 1055, 1060-1062, 1065-1085, 1090-1091, 1117-1121, 1152-1156, 1174, 1177, 1194-1213, 1217-1221, 1229-1230, 1253-1306, 1311->1341, 1313-1323, 1353-1402, 1407, 1408->1410, 1412-1416, 1420-1423, 1468, 1483-1487, 1510, 1514-1518, 1522-1523, 1526-1527, 1531->1543, 1532, 1535, 1538->1531, 1539-1540, 1544-1548, 1552->1558, 1554-1555, 1559->1565, 1561-1562, 1566->1576, 1568->1569, 1571-1572, 1589-1590, 1599-1616, 1632, 1635-1638, 1639->1646, 1641-1643, 1659, 1680, 1683-1687, 1692, 1694-1695, 1700, 1703, 1722-1724, 1728-1729, 1732-1733, 1779-1829, 1847-1848, 1855-1859, 1862-1865, 1887, 1890, 1905-1907, 1910, 1927, 1930, 1934, 1938, 1942-1947, 1978-1980, 1983-2011, 2020, 2057, 2060-2085, 2091-2093, 2131-2170
  - victor/workflows/yaml_to_graph_compiler.py                             386    328    166      0  10.51%   108-112, 204-206, 220-233, 246-416, 427-543, 554-590, 604-642, 654-665, 697-729, 805-809, 815-820, 839-943, 970-1025, 1043-1062, 1089-1090, 1116-1117
  - -----------------------------------------------------------------------------------------------------------------
  - TOTAL                                                               220027 164028  64536   1792  20.88%
  - 30 files skipped due to complete coverage.
  - Coverage HTML written to dir htmlcov
  - Coverage XML written to file coverage.xml
  - =========================== short test summary info ============================
  - FAILED tests/integration/providers/test_ollama_integration.py::test_agent_orchestrator_with_ollama
  - FAILED tests/integration/verticals/test_vertical_plugin_loading.py::TestIntegrationWithBuiltinVerticals::test_external_verticals_listed_with_builtins
  - = 2 failed, 931 passed, 62 skipped, 1633 deselected, 3 xfailed, 791 warnings in 281.04s (0:04:41) =
  - /Users/vijaysingh/code/.venv/lib/python3.12/site-packages/coverage/report_core.py:107: CoverageWarning: Couldn't parse Python file '/Users/vijaysingh/code/codingagent/victor/framework/graph.py' (couldnt-parse); see https://coverage.readthedocs.io/en/7.13.1/messages.html#warning-couldnt-parse
  -   coverage._warn(msg, slug="couldnt-parse")

### ✓ PASS Ruff Linting

- **Duration:** 0.07s
- **Status:** Passed
- **Metrics:**
  - error_count: 0

### ✓ PASS Black Formatting Check

- **Duration:** 1.38s
- **Status:** Passed

### ✗ FAIL Mypy Type Checking

- **Duration:** 64.24s
- **Status:** Failed
- **Metrics:**
  - error_count: 3848
- **Errors:**
  - Found 3848 mypy errors (max: 100)

### ✓ PASS Bandit Security Scan

- **Duration:** 22.68s
- **Status:** Passed

### ✗ FAIL Safety Dependency Check

- **Duration:** 7.88s
- **Status:** Failed

### ✗ FAIL Performance Benchmarks

- **Duration:** 0.14s
- **Status:** Failed
- **Errors:**
  - Benchmark execution failed

### ✗ FAIL Documentation Build

- **Duration:** 79.21s
- **Status:** Failed
- **Errors:**
  - Documentation build failed

### ✓ PASS README Validation

- **Duration:** 0.00s
- **Status:** Passed
- **Metrics:**
  - readme_size: 5009
- **Warnings:**
  - README missing sections: ['Features']

### ✓ PASS Version Validation

- **Duration:** 0.00s
- **Status:** Passed

### ✓ PASS CHANGELOG Validation

- **Duration:** 0.00s
- **Status:** Passed
