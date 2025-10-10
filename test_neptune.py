import neptune

# Ваши данные:
run = neptune.init_run(
    project="antihrist-star/phoenix-v2",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhOTE1NjA0Ni0yMjFjLTRmNDQtYTMzNy04OGQxNjBlMWQzZWIifQ=="
)

run["status"] = "Day 7 setup complete on MSI"
run["computer"] = "MSI"
run["date"] = "2025-10-09"

run.stop()

print("✅ Neptune test successful!")
print("Check: https://app.neptune.ai/antihrist-star/phoenix-v2")