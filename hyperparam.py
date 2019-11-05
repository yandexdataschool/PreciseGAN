from abc import abstractmethod
from random import randint, uniform


class RandArg:
    def __call__(self, *args, **kwargs):
        return self.generate_value()

    @abstractmethod
    def generate_value(self):
        return NotImplementedError


class RandInt(RandArg):
    def __init__(self, lwr=0, upr=1):
        self.lwr = lwr
        self.upr = upr

    def generate_value(self):
        return randint(self.lwr, self.upr)


class RandFloat(RandArg):
    def __init__(self, lwr=0., upr=1.):
        self.lwr = lwr
        self.upr = upr

    def generate_value(self):
        return uniform(self.lwr, self.upr)


class RandNumberLogScale(RandArg):
    def __init__(self, val_bounds, exponent_bounds):
        assert isinstance(val_bounds[0], type(val_bounds[1])), 'val_bound should have the same type'
        if isinstance(val_bounds[0], int):
            self.val_generator = RandInt(val_bounds[0], val_bounds[1])
        else:
            self.val_generator = RandFloat(val_bounds[0], val_bounds[1])
        self.exponent_generator = RandInt(exponent_bounds[0], exponent_bounds[1])

    def generate_value(self):
        return self.val_generator() * (10 ** self.exponent_generator())


class RandChoice(RandArg):
    def __init__(self, select_from):
        self.select_from = select_from
        self.index_generator = RandInt(0, len(select_from) - 1)

    def generate_value(self):
        return self.select_from[self.index_generator()]


def rand_search(main, args, args_generators, tries=1):
    for _ in range(tries):
        for key in args_generators:
            setattr(args, key, args_generators[key]())
        main(args)
