import os
from storage.db_setup import init_db, drop_all_tables
from storage.results_store import save_nlp_result, get_results
from storage.analytics import total_analyses_by_type

TEST_DB = "test_content_analyzer.db"

def setup_module(module):
    """تهيئة قاعدة بيانات تجريبية منعزلة للاختبارات فقط"""
    import storage.db_setup
    import storage.results_store
    import storage.analytics
    storage.db_setup.DB_PATH = TEST_DB
    storage.results_store.DB_PATH = TEST_DB
    storage.analytics.DB_PATH = TEST_DB
    init_db(TEST_DB)

def teardown_module(module):
    """تنظيف وحذف قاعدة البيانات التجريبية بعد الانتهاء"""
    drop_all_tables(TEST_DB)
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)

# 6 & 7 & 8: اختبارات قاعدة البيانات والتحليلات
def test_save_and_get_results():
    saved = save_nlp_result("نص تجريبي", {"dummy": "data"})
    assert saved == True
    results = get_results(type_filter='nlp', limit=1)
    assert len(results) >= 1

def test_analytics_total_analyses():
    save_nlp_result("نص آخر", {"dummy": "data"})
    totals = total_analyses_by_type()
    assert 'nlp' in totals
    assert totals['nlp'] >= 1

def test_empty_results_filter():
    results = get_results(type_filter='unknown_type')
    assert len(results) == 0