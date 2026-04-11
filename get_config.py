import yaml

with open('$YAML_FILE') as f: 
    d=yaml.safe_load(f) 
    print(d)