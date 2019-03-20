import typing
import sympy
import pandas as pd
import toposort
import tensorflow as tf
from io import StringIO
from time import time
from math import inf
from dataclasses import dataclass
from collections import defaultdict
from sympy.physics.units import Dimension, Quantity
from sympy.physics.units.dimensions import dimsys_SI


def si_dimension_scale(expr: sympy.Expr) -> Dimension:
    return Quantity._collect_factor_and_dimension(expr)[0]


def dimension(expr: sympy.Expr) -> Dimension:
    return Quantity._collect_factor_and_dimension(expr)[1]


def assert_same_dimension(expected: sympy.Expr, actual: sympy.Expr):
    expected_dim = dimension(expected)
    actual_dim = dimension(actual)
    assert \
        dimsys_SI.equivalent_dims(expected_dim, actual_dim), \
        f"Incompatible dimensions, expected {expected_dim}, but actually is {actual_dim}"


@dataclass
class Symbol:
    symbol: sympy.Symbol
    description: str
    units: sympy.Expr

    @property
    def name(self) -> str:
        return self.symbol.name

    @property
    def dimension(self) -> Dimension:
        return dimension(self.units)

    @property
    def depends(self) -> typing.List[sympy.Symbol]:
        return []


@dataclass
class Constant(Symbol):
    """
    A known constant.

    E.g., G, the gravitational constant
    """

    value: float

    pass


@dataclass
class Parameter(Symbol):
    """
    A placeholder for output parameters, obtained by fitting the ExperimentalData
    to the target function.

    E.g., on Hooke's law, the parameter is `k`, the sprint constant
    """

    initial_guess: float
    range: typing.Tuple[float, float]

    pass


@dataclass
class ExperimentalData(Symbol):
    """
    A placeholder for input data, obtained via experimental measurements

    E.g., on Hooke's law, the experimental data is `F_exp` and `x`, the measured force and displacement
    """
    pass


@dataclass
class Expression(Symbol):
    """
    Stores the result of a math expression as a named symbol.

    E.g., on Hooke's law, the expression can be `F = k * x`, the predicted force from displacement
    """
    expr: sympy.Expr

    @property
    def depends(self) -> typing.List[sympy.Symbol]:
        return self.expr.free_symbols


@dataclass
class OrdinaryDifferential(Symbol):
    """
    An ODE, in the form `diff(symbol, differential) = expr`
    """
    differential: sympy.Symbol
    expr: sympy.Expr
    initial_value: sympy.Expr

    @property
    def depends(self):
        return (self.expr * self.differential).free_symbols


@dataclass
class OptimizationObjective(Symbol):
    """
    Optimization where we want to minimize the difference between actual and prediction model

    E.g., on Hooke's law, we want to minimize the difference between `F_exp` and `F`,
    the force measured experimentally and the force predicted by the model
    """
    actual: sympy.Expr
    prediction: sympy.Expr
    weight: float

    @property
    def depends(self) -> typing.List[sympy.Symbol]:
        return (self.actual + self.prediction).free_symbols


@dataclass
class _CompositeOrdinaryDifferential(Symbol):
    """
    A set of ODEs with the same differential

    It is handy to integrate multiple ODEs simultaneously, so that we can solve things like this:
    - x'(t) = f1(x(t), y(t), t);
    - y'(t) = f2(x(t), y(t), t);
    """
    differential: sympy.Symbol
    differentials: typing.List[sympy.Expr]

    @property
    def depends(self):
        ret = []
        for ode in self.differentials:
            ret += ode.depends
        return list(set(ret))


class DataFit:
    def __init__(self):
        self.symbols = {}
        self.optimizations = []

    def __getattr__(self, name):
        return self.symbols[name].symbol

    def __getitem__(self, name):
        return self.symbols[name].symbol

    def _add(self, symbol: Symbol):
        self.symbols[symbol.name] = symbol
        return symbol.symbol

    def add_constant(self, name: str, value: float, units: sympy.Expr, description: str):
        return self._add(Constant(
            symbol=sympy.Symbol(name),
            description=description,
            value=value,
            units=units,
        ))

    def add_parameter(self, name: str, units: sympy.Expr, description: str, initial_guess: float = 0, range: typing.Tuple[float, float] = (-inf, +inf)):
        return self._add(Parameter(
            symbol=sympy.Symbol(name),
            description=description,
            units=units,
            initial_guess=initial_guess,
            range=range
        ))

    def add_experimental_data(self, name: str, units: sympy.Expr, description: str):
        return self._add(ExperimentalData(
            symbol=sympy.Symbol(name),
            description=description,
            units=units
        ))

    def add_expr(self, name: str, units: sympy.Expr, expr: sympy.Expr, description: str):
        assert_same_dimension(units, self._units(expr))
        return self._add(Expression(
            symbol=sympy.Symbol(name),
            description=description,
            units=units,
            expr=expr
        ))

    def add_differential_expr(self, name: str, units: sympy.Expr, expr: sympy.Expr, differential: sympy.Symbol, initial_value: sympy.Expr, description: str):
        assert_same_dimension(units, self._units(expr*differential))
        assert_same_dimension(units, self._units(initial_value))
        return self._add(OrdinaryDifferential(
            symbol=sympy.Symbol(name),
            description=description,
            units=units,
            expr=expr,
            differential=differential,
            initial_value=initial_value
        ))

    def add_optimization_objective(self, name: str, units: sympy.Expr, actual: sympy.Expr, prediction: sympy.Expr, weight: float = 1, description: str = 'MSE'):
        assert_same_dimension(units, self._units(actual))
        assert_same_dimension(units, self._units(prediction))
        return self._add(OptimizationObjective(
            symbol=sympy.Symbol(name),
            description=description,
            units=units,
            actual=actual,
            prediction=prediction,
            weight=weight
        ))

    def _units(self, expr):
        return expr.subs([
            (symbol.symbol, symbol.units)
            for symbol in self.symbols.values()
        ])

    def _expr_to_tensor(self, expr: sympy.Expr, result_units: sympy.Expr, tf_symbols: typing.Dict[str, tf.Tensor]):
        # Do unit convertions
        # First, scale all input symbols to SI base units
        expr = expr.subs([
            (s.symbol, s.symbol * si_dimension_scale(s.units))
            for s in self.symbols.values()
        ])
        expr = expr / si_dimension_scale(result_units)  # Convert the result to the desired units

        # Make it into a tf.Tensor
        tf_lambda = sympy.lambdify(expr.free_symbols, expr, "tensorflow")
        return tf_lambda(*[
            tf_symbols[s.name]
            for s in expr.free_symbols
        ])

    def to_tensorflow(self) -> typing.Tuple[tf.Graph, typing.Dict[str, tf.Tensor]]:
        graph = tf.Graph()
        with graph.as_default():
            scope_constants = tf.name_scope('constants/')
            scope_param = tf.name_scope('output_parameters/')
            scope_experimental_data = tf.name_scope('experimental_data/')
            scope_expressions = tf.name_scope('expressions/')
            scope_objectives = tf.name_scope('objectives/')

            all_symbols = dict(self.symbols)
            differentials = {}
            topology_symbols = defaultdict(set)
            for symbol in self.symbols.values():
                if isinstance(symbol, OrdinaryDifferential):
                    if symbol.differential not in differentials:
                        differential = _CompositeOrdinaryDifferential(
                            symbol=sympy.Symbol(f'integral-{symbol.differential.name}'),
                            units=1,  # Not used, and really doesn't make sense here
                            description=f'Simultaneos differential integration on {symbol.differential.name}',
                            differential=symbol.differential,
                            differentials=[]
                        )
                        differentials[symbol.differential] = differential
                        all_symbols[differential.name] = differential
                    differential = differentials[symbol.differential]
                    differential.differentials += [symbol]
                    topology_symbols[differential.name] |= {s.name for s in symbol.depends}
                    topology_symbols[symbol.name] |= {differential.name}
                else:
                    topology_symbols[symbol.name] |= {s.name for s in symbol.depends}

            # The differential calculation is allowed to depend on it's output symbols
            for differential in differentials.values():
                topology_symbols[differential.name] -= {s.name for s in differential.differentials}
            symbols_ordered = [all_symbols[name] for name in toposort.toposort_flatten(topology_symbols)]

            tf_symbols = {}
            tf_objectives = []

            def make_range_constraint(min, max):
                return lambda t: tf.clip_by_value(t, min, max)

            for symbol in symbols_ordered:
                if isinstance(symbol, ExperimentalData):
                    with scope_experimental_data:
                        tf_symbols[symbol.name] = tf.placeholder(
                            name=symbol.name,
                            dtype=tf.float64,
                            shape=(None))

                elif isinstance(symbol, Constant):
                    with scope_constants:
                        tf_symbols[symbol.name] = tf.constant(
                            value=symbol.value,
                            name=symbol.name,
                            dtype=tf.float64)

                elif isinstance(symbol, Parameter):
                    with scope_param:
                        tf_symbols[symbol.name] = tf.Variable(
                            name=symbol.name,
                            dtype=tf.float64,
                            expected_shape=(),
                            initial_value=symbol.initial_guess,
                            constraint=make_range_constraint(symbol.range[0], symbol.range[1]))

                elif isinstance(symbol, Expression):
                    with scope_expressions:
                        with tf.name_scope(symbol.name):
                            tf_symbols[symbol.name] = tf.identity(
                                self._expr_to_tensor(symbol.expr, symbol.units, tf_symbols),
                                name='value')

                elif isinstance(symbol, _CompositeOrdinaryDifferential):
                    with scope_expressions:
                        with tf.name_scope(symbol.name):
                            y0 = tf.stack(
                                [
                                    self._expr_to_tensor(differential.initial_value, differential.units, tf_symbols)
                                    for differential in symbol.differentials
                                ],
                                name='initial_values')

                            def create_func(symbol, tf_symbols):
                                t_symbol = self.symbols[symbol.differential.name]

                                def func(y, t):
                                    replacements = dict(tf_symbols)
                                    replacements[t_symbol.name] = t
                                    replacements.update(dict(zip([x.name for x in symbol.differentials], tf.unstack(y))))

                                    return tf.stack(
                                        [
                                            tf.identity(self._expr_to_tensor(differential.expr, differential.units / t_symbol.units, replacements), name=f'derivative_{differential.name}')
                                            for differential in symbol.differentials
                                        ],
                                        name='derivatives')
                                return func

                            tf_odeint = tf.contrib.integrate.odeint_fixed(
                                func=create_func(symbol, tf_symbols),
                                y0=y0,
                                t=tf.concat([[0], tf_symbols[symbol.differential.name]], axis=0),
                                dt=0.1)
                            tf_symbols[symbol.name] = tf.transpose(tf_odeint[1:, :], name='value')

                elif isinstance(symbol, OrdinaryDifferential):
                    with scope_expressions:
                        with tf.name_scope(symbol.name):
                            composite_symbol = differentials[symbol.differential]
                            composite_index = composite_symbol.differentials.index(symbol)
                            tf_symbols[symbol.name] = tf.identity(
                                tf_symbols[composite_symbol.name][composite_index, :],
                                name='value')

                elif isinstance(symbol, OptimizationObjective):
                    with scope_objectives:
                        with tf.name_scope(symbol.name):
                            tf_actual = tf.identity(
                                self._expr_to_tensor(symbol.actual, symbol.units, tf_symbols),
                                name='actual')
                            tf_prediction = tf.identity(
                                self._expr_to_tensor(symbol.prediction, symbol.units, tf_symbols),
                                name='prediction')
                            tf_loss = tf.losses.mean_squared_error(tf_actual, tf_prediction, weights=symbol.weight)
                            tf_symbols[symbol.name] = tf_loss
                            tf_objectives.append(tf_loss)
                else:
                    raise ValueError(f"Unknown symbol: {symbol}")

            with scope_objectives:
                tf_total_loss = tf.math.add_n(tf_objectives, name='total_loss')

        return graph, tf_symbols, tf_total_loss

    def fit(self, experimental_data: pd.DataFrame, N: int = 1000, v: bool = False, learning_rate: float = 0.1) -> typing.Dict[str, float]:
        graph, tf_symbols, tf_total_loss = self.to_tensorflow()
        with graph.as_default():
            with tf.name_scope('optimizer/'):
                optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
                train_op = optimizer.minimize(tf_total_loss)

            grad_vars = []

            tf_in = {}
            tf_out = []
            tf_out_names = []
            for symbol in self.symbols.values():
                if isinstance(symbol, ExperimentalData):
                    tf_in[tf_symbols[symbol.name]] = experimental_data[symbol.name]
                else:
                    grad_vars.append(symbol.name)

                    tf_out.append(tf_symbols[symbol.name])
                    tf_out_names.append(symbol.name)

            for grad_out in grad_vars:
                grads = tf.gradients(tf_symbols[grad_out], [tf_symbols[grad_in] for grad_in in grad_vars])
                for grad_in, grad in zip(grad_vars, grads):
                    if grad_in != grad_out and grad is not None:
                        print(f"∂{grad_out}/∂{grad_in} = {grad}")
                        tf_out.append(grad)
                        tf_out_names.append(f"∂{grad_out}/∂{grad_in}")

            with tf.Session().as_default() as session:
                session.run(tf.global_variables_initializer())

                print_int = 1
                start_t = time()
                print_t = start_t + print_int

                for i in range(N):
                    now = time()
                    if v and now >= print_t:
                        outputs, loss, _ = session.run([tf_out, tf_total_loss, train_op], feed_dict=tf_in)
                        print("Train state", dict(zip(tf_out_names, outputs)), loss, i)
                        print_t = now + print_int

                    else:
                        session.run([train_op], feed_dict=tf_in)

                outputs = session.run(tf_out, feed_dict=tf_in)
                outputs = dict(zip(tf_out_names, outputs))
                return outputs

def data_csv(data, **kwargs):
    return pd.read_csv(StringIO(data), **kwargs)
def data_tsv(data, **kwargs):
    return data_csv(data, sep='\t', **kwargs)
