import logging
import logging.config


logging.config.dictConfig({
    "version": 1,

    "formatters": {
        "investment_local_horse_racing_predict.logging.format": {
            "format": "%(asctime)s - %(levelname)-5s [%(name)s] %(message)s",
        },
    },

    "handlers": {
        "investment_local_horse_racing_predict.logging.handler": {
            "class": "logging.StreamHandler",
            "formatter": "investment_local_horse_racing_predict.logging.format",
            "level": logging.DEBUG,
        },
    },

    "loggers": {
        "investment_local_horse_racing_predict": {
            "handlers": ["investment_local_horse_racing_predict.logging.handler"],
            "level": logging.DEBUG,
            "propagate": 0,
        },
    },
})


def get_logger(name):
    return logging.getLogger(name)
