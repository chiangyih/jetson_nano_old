from ultralytics import settings

# View all settings
print(settings)

# Return a specific setting
value = settings["runs_dir"]  # Get the directory where runs are saved