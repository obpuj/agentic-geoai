from auie.knowledge import KnowledgeBase
from auie.reporter import report
from auie.llm import default_client
from auie.gazetteer import Gazetteer
from auie.router import Router
from auie.agents.base import StubGISAgent
from auie.agents.vision import VisionAgent
from auie.pipeline import handle_query

gaz = Gazetteer()
router = Router({"vision": VisionAgent(gaz), "gis": StubGISAgent()})
kb = KnowledgeBase.load("data/kb")
client = default_client()

for q in [
    "How much did built-up area grow in Whitefield between 2019 and 2025?",
    "How has Bellandur changed over the years?",
    "Show the city's land cover for 2024.",
]:
    print("=" * 70)
    print("Q:", q)
    res = handle_query(q, client, gaz, router)
    print("status:", res.status)
    if res.status == "ok":
        try:
            answer, cited = report(q, res.bundle, kb, client)
            print("\nANSWER:", answer)
            print("cited chunks:", cited)
        except Exception as e:
            print("\nANSWER: (reporter failed:", type(e).__name__, str(e)[:200], ")")
        print("\nstats:", res.bundle.stats)
        print("provenance:", res.bundle.provenance)
    else:
        print("message:", res.message)





# from auie.knowledge import KnowledgeBase
# from auie.reporter import report
# from auie.llm import default_client
# from auie.gazetteer import Gazetteer
# from auie.router import Router
# from auie.agents.base import StubGISAgent
# from auie.agents.vision import VisionAgent
# from auie.pipeline import handle_query

# gaz = Gazetteer()
# router = Router({"vision": VisionAgent(gaz), "gis": StubGISAgent()})

# kb = KnowledgeBase.load("data/kb")

# for q in [
#     "How much did built-up area grow in Whitefield between 2019 and 2025?",
#     "How has Bellandur changed over the years?",
#     "Show the city's land cover for 2024.",
# ]:
#     print("=" * 70)
#     print("Q:", q)
#     res = handle_query(q, default_client(), gaz, router)
#     print("status:", res.status)
#     if res.status == "ok":
#         answer, cited = report(q, res.bundle, kb, default_client())
#         print("\nANSWER:", answer)
#         print("cited chunks:", cited)
#         print("stats:", res.bundle.stats)
#         print("provenance:", res.bundle.provenance)
#     else:
#         print("message:", res.message)