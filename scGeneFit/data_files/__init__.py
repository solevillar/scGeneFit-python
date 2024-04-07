
import scipy.io
try:
    import importlib.resources as importlib_resources
except ImportError:
    # In PY<3.7 fall-back to backported `importlib_resources`.
    import importlib_resources

def get_data(filename):
    src = importlib_resources.files(__name__).joinpath(filename)
    with importlib_resources.as_file(src) as foo:
        return scipy.io.loadmat(str(foo))
