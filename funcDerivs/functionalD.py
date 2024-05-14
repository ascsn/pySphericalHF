"""Evaluate the functional derivative for a given edf"""

import numpy as np
import sympy as sp
import sympy.printing as printing
from sympy import init_session
from sympy.core.function import AppliedUndef
from sympy import DiracDelta
import sympy.physics as sphys
from sympy.vector import CoordSys3D, Del
from sympy.abc import t


init_session(quiet=True)
sp.init_printing(use_latex=True)

# Universal constants
hbar = 197.327  # MeV-fm
m = 940  # MeV


class EDF:
    def __init__(self, var, inputFunc, *funcs, **kwargs):
        """
        EDF class that derives all properties of the given EDF.

        Parameters:
        ----------
        var: list
            list of variables

        inputFunc: string/function
            EDF functional

        funcs: string
            string of functionals that EDF depends on

        kwargs: dictionary
            a dictionary of coefficients and their values/definitions

        Returns:
        None.

        :Example:
            Let's say the EDF is edfFunction = a0 + a1*rho_n + a2*rho_n*rho_p + a3*rho_0 and
            rho_n depends on rho_0 as rho_n = rho_0/2
            var = ["rho_n", "rho_p"]
            funcs = "rho_0/2" where rho_0 = sp.symbols('rho_0', cls = sp.Function)(rho_n)
            dictOfCoeffs = {"a0":a0, "a1":a1, "a2":a2, "a3":a3}
            EDF(var, edfFunction, *funcs, **dictOfCoeffs)
        """
        if kwargs:
            self.__dict__.update(kwargs)
        if isinstance(var, list):
            var = ",".join(var)
        self.var = sp.symbols(str(var))
        self.rho0 = None
        self.funcs = None
        if funcs:
            if isinstance(funcs[0], list):
                self.funcs = funcs[0]
            else:
                self.funcs = list[funcs[0]]
        self.edf_fun(inputFunc)
        self.deriv = self.functional
        self.count = 0

    def __repr__(self) -> str:
        return f"EDF({self.var},{self.functional}, {self.funcs}, {self.__dict__})"

    @staticmethod
    def gradient(scalar_function, variables):
        "Use this gradient instead of the inbuilt one as it is easier to work with."
        if not isinstance(variables, list):
            variables = list(variables)
        grad_f = [sp.diff(scalar_function, var) for var in variables]
        return sp.Matrix(grad_f)

    @staticmethod
    def divergence(grad_f, vars):
        "Use this divergence function instead of the inbuilt one as it's easier to work with."
        div_grad_f = sum(
            sp.diff(component, var) for component, var in zip(grad_f, vars)
        )
        return div_grad_f

    def edf_fun(self, inputFunc):
        """
        Reads in the EDF functional and converts it to sympy function.
        :Example:
            EDF.edf_fun(edfFunction)

        Parameters:
        -----------
        inputFunc: string
            A function that is converted to sympy function for further use.

        Returns:
        --------
        functional: sympy function
            the object functional to evaluate all the properties.
        """
        self.functional = sp.sympify(inputFunc)
        return self.functional

    def grad_derivative(self, *args):
        outputFinal = []
        y = sp.symbols("y")
        try:
            argsList = [args[0].components[key] for key in args[0].components]
        except AttributeError:
            print("Attribute Error: Not Implemented")
            return None

        def process_factors(values):
            if isinstance(values, sp.core.function.Derivative):
                newFunc = self.deriv.replace(values, y)
                output = sp.diff(newFunc, y)
                return output.replace(y, values)

        try:
            for val in argsList:
                output = process_factors(val)
                outputFinal.append(output)
        except:
            for val in argsList:
                factors = sp.expand(val).args
                for values in factors:
                    newFactors = sp.expand(values).args
                    output = process_factors(newFactors)
                    outputFinal.append(output)
        return sum(outputFinal)

    def derivative(self, order=1, *args):
        """
        The general derivative of the given EDF functional

        :Example:
        EDF.derivative(order=n)
            returns n-th derivative wrt all the variables

        Parameters:
        -----------
        order: integer, optional
            Order of the derivative

        args:
            The derivative wrt the variables provided

        Returns:
        --------
        diff: The n-th order derivative wrt all (or supplied) the variables of the EDF.

        """
        if self.count == 0:
            self.deriv = self.functional
        if order < 1:
            return ValueError("Order must be >= 1")
        if order == 1:
            if args:
                l1 = [
                    sp.vector.vector.VectorMul,
                    sp.vector.vector.VectorAdd,
                    sp.core.power.Pow,
                ]
                if callable(args[0]) or any(isinstance(args[0], A) for A in l1):
                    return self.grad_derivative(*args)
                output = sp.derive_by_array(self.deriv, args[0]).simplify()
                self.count = 0
                self.deriv = self.functional
                return output
            if self.funcs:
                output = sp.derive_by_array(self.deriv, self.funcs).simplify()
                self.count = 0
                self.deriv = self.functional
                return output
            output = sp.derive_by_array(self.deriv, self.var).simplify()
            self.count = 0
            self.deriv = self.functional
            return output
        else:
            if args:
                if callable(args[0]) or isinstance(args[0], sp.vector.vector.VectorMul):
                    self.deriv = self.grad_derivative(*args)
                    self.count += 1
                    return self.derivative(order - 1, *args)
                self.deriv = sp.derive_by_array(self.deriv, args[0]).simplify()
                self.count += 1
                return self.derivative(order - 1, *args)
            if self.funcs:
                self.deriv = sp.derive_by_array(self.deriv, self.funcs).simplify()
                self.count += 1
                return self.derivative(order - 1, *args)
            self.deriv = sp.derive_by_array(self.deriv, self.var).simplify()
            self.count += 1
            return self.derivative(order - 1, *args)
