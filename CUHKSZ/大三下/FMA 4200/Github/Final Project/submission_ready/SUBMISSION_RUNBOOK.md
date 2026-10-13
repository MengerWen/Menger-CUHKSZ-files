# Submission Runbook

## Full Rebuild

```powershell
& 'd:\MG\anaconda3\python.exe' 'scripts\check_environment.py'
& 'd:\MG\anaconda3\python.exe' 'scripts\run_pipeline.py'
& 'd:\MG\anaconda3\python.exe' 'scripts\audit_and_prepare_submission.py'
```

## Fast Report-and-Submission Refresh

```powershell
& 'd:\MG\anaconda3\python.exe' 'scripts\build_final_report.py'
& 'd:\MG\anaconda3\python.exe' 'scripts\audit_and_prepare_submission.py'
```

## Main Deliverables After the Audit Step

- `report/final_report.md`
- `report/final_report.pdf`
- `report/final_appendices.md`
- `report/rubric_checklist.md`
- `report/audit_report.md`
- `submission_ready/`
- `submission_ready.zip`
