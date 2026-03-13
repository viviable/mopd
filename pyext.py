import importlib.util
import sys
import types


class RuntimeModule:
    @staticmethod
    def from_string(module_name, module_path, source):
        module = types.ModuleType(module_name)
        module.__file__ = module_path or f"<{module_name}>"
        module.__package__ = ""

        spec = importlib.util.spec_from_loader(module_name, loader=None)
        if spec is not None:
            module.__spec__ = spec

        sys.modules[module_name] = module
        exec(compile(source, module.__file__, "exec"), module.__dict__)
        return module
