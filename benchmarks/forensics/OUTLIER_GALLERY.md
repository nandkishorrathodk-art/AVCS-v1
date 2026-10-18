# AVCS Outlier Error Gallery (Top 16 Long-Tail Errors)

Visual diagnostics showing Ground Truth (Green box/dot), Arm A (Blue cross), Arm B (Orange cross), and Arm C (Red cross).

| Rank | Query | Category | Arm A Err | Arm B Err | In Box? | Failure Diagnosis |
|---|---|---|---|---|---|---|
| #1 | **Files icon** | `desktop_canvas` | 79.84 px | **78.67 px** | False | **AMBIGUOUS_QUERY_OR_INDISTINGUISHABLE_CANDIDATES**: True target was detected as candidate peak #2-#4, but another candidate won top rank. |
| #2 | **Projects icon** | `desktop_canvas` | 76.49 px | **75.96 px** | False | **AMBIGUOUS_QUERY_OR_INDISTINGUISHABLE_CANDIDATES**: True target was detected as candidate peak #2-#4, but another candidate won top rank. |
| #3 | **Edit button** | `editor_code` | 36.13 px | **35.4 px** | False | **AMBIGUOUS_QUERY_OR_INDISTINGUISHABLE_CANDIDATES**: True target was detected as candidate peak #2-#4, but another candidate won top rank. |
| #4 | **View button** | `editor_code` | 32.74 px | **32.88 px** | False | **AMBIGUOUS_QUERY_OR_INDISTINGUISHABLE_CANDIDATES**: True target was detected as candidate peak #2-#4, but another candidate won top rank. |
| #5 | **Run button** | `editor_code` | 32.59 px | **32.4 px** | False | **AMBIGUOUS_QUERY_OR_INDISTINGUISHABLE_CANDIDATES**: True target was detected as candidate peak #2-#4, but another candidate won top rank. |
| #6 | **Tools button** | `editor_code` | 27.7 px | **27.44 px** | False | **AMBIGUOUS_QUERY_OR_INDISTINGUISHABLE_CANDIDATES**: True target was detected as candidate peak #2-#4, but another candidate won top rank. |
| #7 | **Find... input field** | `editor_code` | 25.51 px | **26.4 px** | True | **CORRECT_TARGET_REGION_INACCURATE_CENTER**: Prediction is inside the 134.5x17.9px bounding box (valid click), but offset 26.4px from annotated geometric center. |
| #8 | **team@visualcortex.ai input field** | `form_input` | 23.33 px | **22.2 px** | False | **AMBIGUOUS_QUERY_OR_INDISTINGUISHABLE_CANDIDATES**: True target was detected as candidate peak #2-#4, but another candidate won top rank. |
| #9 | **Tools button** | `editor_code` | 22.15 px | **21.92 px** | False | **AMBIGUOUS_QUERY_OR_INDISTINGUISHABLE_CANDIDATES**: True target was detected as candidate peak #2-#4, but another candidate won top rank. |
| #10 | **Search Google or type URL input field** | `browser_webpage` | 20.44 px | **20.44 px** | True | **CORRECT_TARGET_REGION_INACCURATE_CENTER**: Prediction is inside the 168.3x22.9px bounding box (valid click), but offset 20.4px from annotated geometric center. |
| #11 | **I'm Feeling Lucky button** | `browser_webpage` | 17.11 px | **17.95 px** | True | **CORRECT_TARGET_REGION_INACCURATE_CENTER**: Prediction is inside the 95.6x19.9px bounding box (valid click), but offset 17.9px from annotated geometric center. |
| #12 | **Apply Changes button** | `settings_panel` | 17.75 px | **17.04 px** | True | **CORRECT_TARGET_REGION_INACCURATE_CENTER**: Prediction is inside the 135.5x15.9px bounding box (valid click), but offset 17.0px from annotated geometric center. |
| #13 | ********** input field** | `form_input` | 17.86 px | **16.96 px** | True | **CORRECT_TARGET_REGION_INACCURATE_CENTER**: Prediction is inside the 221.1x19.9px bounding box (valid click), but offset 17.0px from annotated geometric center. |
| #14 | **Enter command... input field** | `terminal_shell` | 15.63 px | **15.44 px** | True | **CORRECT_TARGET_REGION_INACCURATE_CENTER**: Prediction is inside the 224.1x18.9px bounding box (valid click), but offset 15.4px from annotated geometric center. |
| #15 | **Enter command... input field** | `terminal_shell` | 15.41 px | **14.95 px** | True | **CORRECT_TARGET_REGION_INACCURATE_CENTER**: Prediction is inside the 229.1x18.9px bounding box (valid click), but offset 14.9px from annotated geometric center. |
| #16 | **Search... input field** | `settings_panel` | 13.78 px | **12.98 px** | True | **CORRECT_TARGET_REGION_INACCURATE_CENTER**: Prediction is inside the 227.1x17.9px bounding box (valid click), but offset 13.0px from annotated geometric center. |


## Visual Crops

### Rank #1: Files icon (`desktop_canvas`)
- **Error (Arm B):** 78.67 px (In Box: `False`)
- **Category:** `AMBIGUOUS_QUERY_OR_INDISTINGUISHABLE_CANDIDATES`
- **Diagnosis:** True target was detected as candidate peak #2-#4, but another candidate won top rank.

![Outlier #1](gallery/outlier_01_screen_test_held_out_0004_Files_icon.png)

---

### Rank #2: Projects icon (`desktop_canvas`)
- **Error (Arm B):** 75.96 px (In Box: `False`)
- **Category:** `AMBIGUOUS_QUERY_OR_INDISTINGUISHABLE_CANDIDATES`
- **Diagnosis:** True target was detected as candidate peak #2-#4, but another candidate won top rank.

![Outlier #2](gallery/outlier_02_screen_test_held_out_0004_Projects_icon.png)

---

### Rank #3: Edit button (`editor_code`)
- **Error (Arm B):** 35.4 px (In Box: `False`)
- **Category:** `AMBIGUOUS_QUERY_OR_INDISTINGUISHABLE_CANDIDATES`
- **Diagnosis:** True target was detected as candidate peak #2-#4, but another candidate won top rank.

![Outlier #3](gallery/outlier_03_screen_test_held_out_0010_Edit_button.png)

---

### Rank #4: View button (`editor_code`)
- **Error (Arm B):** 32.88 px (In Box: `False`)
- **Category:** `AMBIGUOUS_QUERY_OR_INDISTINGUISHABLE_CANDIDATES`
- **Diagnosis:** True target was detected as candidate peak #2-#4, but another candidate won top rank.

![Outlier #4](gallery/outlier_04_screen_test_held_out_0009_View_button.png)

---

### Rank #5: Run button (`editor_code`)
- **Error (Arm B):** 32.4 px (In Box: `False`)
- **Category:** `AMBIGUOUS_QUERY_OR_INDISTINGUISHABLE_CANDIDATES`
- **Diagnosis:** True target was detected as candidate peak #2-#4, but another candidate won top rank.

![Outlier #5](gallery/outlier_05_screen_test_held_out_0009_Run_button.png)

---

### Rank #6: Tools button (`editor_code`)
- **Error (Arm B):** 27.44 px (In Box: `False`)
- **Category:** `AMBIGUOUS_QUERY_OR_INDISTINGUISHABLE_CANDIDATES`
- **Diagnosis:** True target was detected as candidate peak #2-#4, but another candidate won top rank.

![Outlier #6](gallery/outlier_06_screen_test_held_out_0008_Tools_button.png)

---

### Rank #7: Find... input field (`editor_code`)
- **Error (Arm B):** 26.4 px (In Box: `True`)
- **Category:** `CORRECT_TARGET_REGION_INACCURATE_CENTER`
- **Diagnosis:** Prediction is inside the 134.5x17.9px bounding box (valid click), but offset 26.4px from annotated geometric center.

![Outlier #7](gallery/outlier_07_screen_test_held_out_0006_Find____input_field.png)

---

### Rank #8: team@visualcortex.ai input field (`form_input`)
- **Error (Arm B):** 22.2 px (In Box: `False`)
- **Category:** `AMBIGUOUS_QUERY_OR_INDISTINGUISHABLE_CANDIDATES`
- **Diagnosis:** True target was detected as candidate peak #2-#4, but another candidate won top rank.

![Outlier #8](gallery/outlier_08_screen_test_held_out_0029_team_visualcortex_ai_inpu.png)

---

### Rank #9: Tools button (`editor_code`)
- **Error (Arm B):** 21.92 px (In Box: `False`)
- **Category:** `AMBIGUOUS_QUERY_OR_INDISTINGUISHABLE_CANDIDATES`
- **Diagnosis:** True target was detected as candidate peak #2-#4, but another candidate won top rank.

![Outlier #9](gallery/outlier_09_screen_test_held_out_0006_Tools_button.png)

---

### Rank #10: Search Google or type URL input field (`browser_webpage`)
- **Error (Arm B):** 20.44 px (In Box: `True`)
- **Category:** `CORRECT_TARGET_REGION_INACCURATE_CENTER`
- **Diagnosis:** Prediction is inside the 168.3x22.9px bounding box (valid click), but offset 20.4px from annotated geometric center.

![Outlier #10](gallery/outlier_10_screen_test_held_out_0014_Search_Google_or_type_URL.png)

---

### Rank #11: I'm Feeling Lucky button (`browser_webpage`)
- **Error (Arm B):** 17.95 px (In Box: `True`)
- **Category:** `CORRECT_TARGET_REGION_INACCURATE_CENTER`
- **Diagnosis:** Prediction is inside the 95.6x19.9px bounding box (valid click), but offset 17.9px from annotated geometric center.

![Outlier #11](gallery/outlier_11_screen_test_held_out_0013_I_m_Feeling_Lucky_button.png)

---

### Rank #12: Apply Changes button (`settings_panel`)
- **Error (Arm B):** 17.04 px (In Box: `True`)
- **Category:** `CORRECT_TARGET_REGION_INACCURATE_CENTER`
- **Diagnosis:** Prediction is inside the 135.5x15.9px bounding box (valid click), but offset 17.0px from annotated geometric center.

![Outlier #12](gallery/outlier_12_screen_test_held_out_0036_Apply_Changes_button.png)

---

### Rank #13: ******** input field (`form_input`)
- **Error (Arm B):** 16.96 px (In Box: `True`)
- **Category:** `CORRECT_TARGET_REGION_INACCURATE_CENTER`
- **Diagnosis:** Prediction is inside the 221.1x19.9px bounding box (valid click), but offset 17.0px from annotated geometric center.

![Outlier #13](gallery/outlier_13_screen_test_held_out_0030__________input_field.png)

---

### Rank #14: Enter command... input field (`terminal_shell`)
- **Error (Arm B):** 15.44 px (In Box: `True`)
- **Category:** `CORRECT_TARGET_REGION_INACCURATE_CENTER`
- **Diagnosis:** Prediction is inside the 224.1x18.9px bounding box (valid click), but offset 15.4px from annotated geometric center.

![Outlier #14](gallery/outlier_14_screen_test_held_out_0018_Enter_command____input_fi.png)

---

### Rank #15: Enter command... input field (`terminal_shell`)
- **Error (Arm B):** 14.95 px (In Box: `True`)
- **Category:** `CORRECT_TARGET_REGION_INACCURATE_CENTER`
- **Diagnosis:** Prediction is inside the 229.1x18.9px bounding box (valid click), but offset 14.9px from annotated geometric center.

![Outlier #15](gallery/outlier_15_screen_test_held_out_0019_Enter_command____input_fi.png)

---

### Rank #16: Search... input field (`settings_panel`)
- **Error (Arm B):** 12.98 px (In Box: `True`)
- **Category:** `CORRECT_TARGET_REGION_INACCURATE_CENTER`
- **Diagnosis:** Prediction is inside the 227.1x17.9px bounding box (valid click), but offset 13.0px from annotated geometric center.

![Outlier #16](gallery/outlier_16_screen_test_held_out_0038_Search____input_field.png)

---

