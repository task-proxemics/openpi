import os, sys, subprocess, importlib, traceback
from pathlib import Path

ROOT = Path('/workspace/openpi_jax').resolve()
SRC  = ROOT / 'src'
sys.path[:0] = [str(ROOT), str(SRC)]
os.environ.setdefault('OPENPI_FORCE_BACKEND', 'jax')

MAP = {
  'cv2': 'opencv-python-headless==4.10.0.84',
  'PIL': 'pillow>=11.0.0',
  'yaml': 'pyyaml',
  'sklearn': 'scikit-learn',
  'huggingface_hub': 'huggingface-hub>=0.23.0',
  'dm_tree': 'dm-tree==0.1.8',
  'tree': 'dm-tree==0.1.8',
  'orbax': 'orbax-checkpoint==0.11.13',
  'websockets': 'websockets>=11.0.0',
  'pynvml': 'nvidia-ml-py3',
  'numpydantic': 'numpydantic==1.6.6',
  'beartype': 'beartype==0.19.0',
  'polars': 'polars',
  'regex': 'regex',
  'absl': 'absl-py',
  # do not touch JAX/Numpy in this image:
  'jax': None, 'jaxlib': None, 'numpy': None, 'scipy': None,
}

def pip_install(spec: str):
  print(f'  → pip install {spec}')
  subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', spec], check=True)

def patch_normalize():
  p = ROOT / 'src/openpi/shared/normalize.py'
  if not p.exists(): return
  t = p.read_text()
  if 'OPENPI_USE_PYDANTIC' in t and '_BaseModelMixin' in t: return
  import re
  preface = '''
import os
from dataclasses import dataclass as _py_dataclass
from typing import TYPE_CHECKING
try:
    import pydantic  # type: ignore
except Exception:
    pydantic = None  # type: ignore
Dataclass = (
    pydantic.dataclasses.dataclass if (pydantic is not None and os.environ.get("OPENPI_USE_PYDANTIC","0") == "1")
    else _py_dataclass
)
class _BaseModelMixin:
    def __init__(self, **data):
        for k, v in data.items(): setattr(self, k, v)
    @classmethod
    def model_validate(cls, v):
        if isinstance(v, cls): return v
        if isinstance(v, dict): return cls(**v)
        return cls(**dict(v))
    def model_dump(self):
        ann = getattr(self, '__annotations__', {})
        return {k: getattr(self, k) for k in ann if hasattr(self, k)} if ann else dict(self.__dict__)
if TYPE_CHECKING:
    from numpydantic import NDArray  # type: ignore
else:
    try:
        from numpy.typing import NDArray  # type: ignore
    except Exception:
        class NDArray:  # type: ignore
            pass
'''.lstrip()
  t = preface + t
  t = t.replace('@pydantic.dataclasses.dataclass', '@Dataclass')
  t = t.replace('(pydantic.BaseModel):', '(_BaseModelMixin):')
  t = re.sub(r'^\s*from\s+numpydantic\s+import\s+NDArray\s*$',
             'from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from numpydantic import NDArray  # type: ignore',
             t, flags=re.MULTILINE)
  Path(str(p)+'.bak').write_text(p.read_text())
  p.write_text(t)
  print('[patched] normalize.py')

def import_sweep():
  tried = set()
  while True:
    try:
      importlib.invalidate_caches()
      __import__('scripts.serve_policy')
      print('✅ serve_policy import OK')
      return
    except ModuleNotFoundError as e:
      name = (e.name or '').split('.')[0]
      if not name: raise
      if name in tried:
        raise
      tried.add(name)
      spec = MAP.get(name, name)
      if spec is None:
        print(f'  → Skipping {name} (managed by base image)')
      else:
        pip_install(spec)
        continue
    except Exception as e:
      tb = ''.join(traceback.format_exception_only(type(e), e))
      if ('InvalidSchemaError' in tb or
          'MissingDefinitionError' in tb or
          'numpydantic' in tb):
        patch_normalize()
        continue
      raise

if __name__ == '__main__':
  import_sweep()
