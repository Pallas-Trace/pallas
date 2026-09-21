# Developers documentation

## Upgrading the Pypy package

### API Token
Create an API token at https://pypi.org/manage/account/token/
[pypi]
username = __token__
password = pypi-xxxxxxxx...

### Installing dependencies
```
apt install patchelf
pip install build twine auditwheel
```

### Publishing

```
python -m build
auditwheel repair dist/pallas_trace-*-linux_x86_64.whl 
rm dist/pallas_trace-*-linux_x86_64.whl 
mv wheelhouse/pallas_trace-0.21-cp314-cp314-manylinux_2_39_x86_64.whl dist/
twine upload dist/*
```
