"""Русскоязычная справка и ошибки командной строки без переименования параметров."""

import argparse
import sys


class RussianArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._positionals.title = "Позиционные параметры"
        self._optionals.title = "Параметры"
        for action in self._actions:
            if isinstance(action, argparse._HelpAction):
                action.help = "Показать справку и завершить работу"

    def format_usage(self):
        return super().format_usage().replace("usage:", "Использование:", 1)

    def format_help(self):
        return super().format_help().replace("usage:", "Использование:", 1)

    def error(self, message):
        translations = {
            "the following arguments are required:": "нужно указать обязательные параметры:",
            "unrecognized arguments:": "неизвестные параметры:",
            "invalid choice:": "недопустимое значение:",
            "choose from": "допустимые значения:",
            "expected at least one argument": "нужно указать хотя бы одно значение",
            "expected one argument": "нужно указать одно значение",
            "expected at most one argument": "можно указать не более одного значения",
            "invalid int value": "недопустимое целое число",
            "invalid float value": "недопустимое вещественное число",
            "argument ": "параметр ",
        }
        for original, translated in translations.items():
            message = message.replace(original, translated)
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: ошибка: {message}\n")
