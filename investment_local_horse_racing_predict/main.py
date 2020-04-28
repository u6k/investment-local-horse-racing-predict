from flask import Flask

from investment_local_horse_racing_predict.app_logging import get_logger


logger = get_logger(__name__)


app = Flask(__name__)


@app.route("/api/health")
def health():
    logger.info("#health: start")

    return "ok"
