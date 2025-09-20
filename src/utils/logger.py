import logging


class DebugLogger:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.logger = logging.getLogger('BibleStylePipeline')

        if debug:
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
        else:
            logging.basicConfig(level=logging.INFO)

    def print(self, message: str, level: str = "info"):
        if self.debug:
            if level == "debug":
                self.logger.debug(message)
            elif level == "info":
                self.logger.info(message)
            elif level == "warning":
                self.logger.warning(message)
            elif level == "error":
                self.logger.error(message)
        else:
            if level != "debug":
                if level == "info":
                    self.logger.info(message)
                elif level == "warning":
                    self.logger.warning(message)
                elif level == "error":
                    self.logger.error(message)

    def print_cross_validation_results(self, results: dict, model_name: str = "Modelo") -> None:
        print(f"\n{'=' * 50}")
        print(f"results cross validation model: {model_name.upper()}")
        print(f"{'=' * 50}")

        print(f"mean accuracy: {results['mean_accuracy']:.4f} (±{results['std_accuracy']:.4f})")
        print(f"mean f1: {results['mean_f1']:.4f} (±{results['std_f1']:.4f})")

        if self.debug:
            print(f"\ndetails per fold")
            for i, fold_metrics in enumerate(results['fold_metrics'], 1):
                print(f"   Fold {i}: accuracy = {fold_metrics['accuracy']:.4f}, f1 = {fold_metrics['f1']:.4f}")


global_logger = DebugLogger(True)


def set_global_debug_mode(debug: bool):
    global global_logger
    global_logger = DebugLogger(debug=debug)


def print_debug(message: str):
    global_logger.print(message, "debug")


def print_info(message: str):
    global_logger.print(message, "info")


def print_warning(message: str):
    global_logger.print(message, "warning")


def print_error(message: str):
    global_logger.print(message, "error")


def print_cross_validation_results(results: dict, model_name: str = "Modelo"):
    global_logger.print_cross_validation_results(results, model_name)
