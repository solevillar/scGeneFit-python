
try:
    import importlib.resources as importlib_resources
except ImportError:
    # In PY<3.7 fall-back to backported `importlib_resources`.
    import importlib_resources

def get_data(filename):
    with importlib_resources.path(__name__, filename) as foo:
        return str(foo)