import pathlib
import pytest


TEST_DIR = "../testdata"
    
@pytest.fixture
def testdata_dir(request):
    # Get the path of the current test file
    test_file = request.node.nodeid
    test_srcdir = pathlib.Path(test_file).parent
    return test_srcdir / TEST_DIR
