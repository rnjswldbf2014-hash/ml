# dplayg

D언어로 구현한 RL/SL 라이브러리. `my_ml.d` → `my_ml.pyd` (.pyd = Python extension, `import my_ml`로 사용).
`my_ml.py`(PyTorch 버전)와 API 동일, `.pth` 파일 포맷 호환.

## 빌드

```powershell
$ldc   = "ldc2\ldc2-1.42.0-windows-x64\bin\ldc2.exe"
$pylib = "$env:LOCALAPPDATA\Programs\Python\Python313\libs\python313.lib"
& $ldc my_ml.d $pylib --O3 --release --shared --link-defaultlib-shared=false -of=my_ml.pyd
Remove-Item my_ml.obj, my_ml.lib -ErrorAction SilentlyContinue
```

## 파일

- `my_ml.d` — D 소스 (신경망 + Python C API + Python 클래스 코드 인라인)
- `my_ml.py` — 원본 PyTorch 버전
- `my_ml.pyd` — 빌드 결과물 (gitignore)
