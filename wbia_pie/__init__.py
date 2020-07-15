from wbia_pie import _plugin  # NOQA
from wbia_pie import __main__  # NOQA

try:
    from wbia_pie._version import __version__
except ImportError:
    __version__ = '0.0.0'
