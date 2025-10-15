from src.config import Config
print('HISTORY_FILE:', Config.HISTORY_FILE)
# Save test history
hist = [{'id':'1','question':'test'}]
print('Saving history ->', Config.save_query_history(hist))
print('Loaded history ->', Config.get_query_history())
# Clear history
print('Clearing ->', Config.save_query_history([]))
print('After clear ->', Config.get_query_history())
