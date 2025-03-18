from typing import Generator, Tuple
from typing import Dict, Any, Optional, Sequence, Tuple


class Module:
    """Modules form a tree that store parameters and other
    submodules. They make up the basis of neural network stacks.

    Attributes
    ----------
        _modules : Storage of the child modules
        _parameters : Storage of the module's parameters
        training : Whether the module is in training mode or evaluation mode
    """

    _modules: Dict[str, "Module"]
    _parameters: Dict[str, "Parameter"]
    training: bool

    def __init__(self) -> None:
        """Initialize a module, setting up storage for child modules and parameters."""
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self) -> Sequence["Module"]:
        """Return the direct child modules of this module."""
        m: Dict[str, Module] = self.__dict__["_modules"]
        return list(m.values())

    def train(self) -> None:
        """Set the mode of this module and all descendant modules to `train`."""
        self.training = True
        # Recursively set all child modules to training mode
        for module in self._modules.values():
            module.train()

    def eval(self) -> None:
        """Set the mode of this module and all descendant modules to `eval`."""
        self.training = False
        # Recursively set all child modules to evaluation mode
        for module in self._modules.values():
            module.eval()

    def named_parameters(self) -> Sequence[Tuple[str, "Parameter"]]:
        """Collect all the parameters of this module and its descendants.

        Returns
        -------
            The name and `Parameter` of each ancestor parameter.
        """
        params = [(key, param) for key, param in self._parameters.items()]
        for module_name, module in self._modules.items():
            for param_name, param in module.named_parameters():
                # Prefix the name with the module name to make it unique
                params.append((f"{module_name}.{param_name}", param))
        return params

    def parameters(self) -> Generator["Parameter", None, None]:
        """Enumerate over all the parameters of this module and its descendants."""
        # Yield parameters of this module
        for param in self._parameters.values():
            yield param
        # Recursively yield parameters from child modules
        for module in self._modules.values():
            yield from module.parameters()

    def add_parameter(self, k: str, v: Any) -> "Parameter":
        """Manually add a parameter. Useful helper for scalar parameters.

        Args:
        ----
            k: Local name of the parameter.
            v: Value for the parameter.

        Returns
        -------
            Newly created parameter.
        """
        val = Parameter(v, k)
        self.__dict__["_parameters"][k] = val
        return val

    def __setattr__(self, key: str, val: "Parameter") -> None:
        """Override the default setter for attributes to handle Parameters and Modules."""
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Any:
        """Override the default getter to return parameters or modules by name."""
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Allow the module to be called as a function (e.g., `module()`).

        This should trigger the forward pass of the module.
        """
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        """Generate a string representation of the module and its submodules."""
        def _addindent(s_: str, numSpaces: int) -> str:
            """Helper function to add indentation for nested submodules."""
            s2 = s_.split("\n")
            if len(s2) == 1:
                return s_
            first = s2.pop(0)
            s2 = [(numSpaces * " ") + line for line in s2]
            s = "\n".join(s2)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class Parameter:
    """A Parameter is a special container stored in a `Module`.

    It is designed to hold a `Variable`, but we allow it to hold
    any value for testing.
    """

    def __init__(self, x: Any, name: Optional[str] = None) -> None:
        """Initialize a parameter with a value and an optional name."""
        self.value = x
        self.name = name
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def update(self, x: Any) -> None:
        """Update the parameter value."""
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def __repr__(self) -> str:
        """Return the string representation of the parameter."""
        return repr(self.value)

    def __str__(self) -> str:
        """Return the string form of the parameter."""
        return str(self.value)
