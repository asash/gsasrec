import tornado.ioloop
import tornado.web
import json
import os

from aprec.ui.config import CATALOG, RECOMMENDER


class SearchHandler(tornado.web.RequestHandler):
    def get(self):
        keyword = self.request.arguments.get("keyword")[0].decode("utf-8")
        items = CATALOG.search(keyword)
        result = []
        for item in items:
            result.append("[{}]  {}".format(item.item_id, item.title))
        self.set_header('Content-Type', 'application/json')
        self.write(json.dumps(result, indent=4))


class RecommenderHandler(tornado.web.RequestHandler):
    def post(self):
        history_raw = [item.decode("utf-8") for item in self.request.arguments['history[]']]
        history_item_ids = [item.split("]")[0].strip("[") for item in history_raw]
        recommendations = RECOMMENDER.recommend_by_items(history_item_ids, 10)
        result = []
        for item in recommendations:
            result.append("[{}] {}".format(item[0], CATALOG.get_item(item[0]).title))
        self.set_header('Content-Type', 'application/json')
        self.write(json.dumps(result, indent=4))


def make_app():
    current_dir = os.path.dirname(__file__)
    static_dir = os.path.join(current_dir, "static")
    print(static_dir)
    return tornado.web.Application([
        (r"/search", SearchHandler),
        (r"/recommend", RecommenderHandler),
        (r"/(.*)", tornado.web.StaticFileHandler, {"path": static_dir, "default_filename": "index.html"})
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(31337)
    tornado.ioloop.IOLoop.current().start()
