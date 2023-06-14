
from transformers.models.mask2former import *
import sys

# Get all imported and present module names
imported_modules = [name for name in dir() if isinstance(eval(name), type(sys))]
print("Imported modules:")
print(imported_modules)

# Get all imported and present function names
imported_functions = [name for name in dir() if callable(eval(name))]
print("\nImported functions:")
print(imported_functions)

# Get all imported and present class names
imported_classes = [name for name in dir() if isinstance(eval(name), type)]
print("\nImported classes:")
print(imported_classes)
