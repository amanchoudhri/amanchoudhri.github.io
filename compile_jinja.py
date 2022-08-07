"""Compile the content.yaml file to HTML."""

from ruamel.yaml import YAML

from jinja2 import Template

from config import HTML_INDEX, JINJA_TEMPLATE, YAML_FILE, SECTIONS

yaml = YAML(typ='safe')

def load_yaml(file=YAML_FILE) -> dict:
    """Load and return the given YAML file. By default, load content.yaml."""
    with open(file, 'r') as file:
        content = yaml.load(file)
    return content

def jinjafy_experience(e):
    if e.get('description'):
        paragraphs = e['description'].split('\n')
        main_text = ''.join([
            f'<p>{p}</p>' for p in paragraphs
        ])
        e['description'] = main_text
    return e

def compile_jinja(template=JINJA_TEMPLATE, index=HTML_INDEX):
    content = load_yaml()

    with open(template, 'r') as f:
        base = Template(
            f.read()
        )
    
    params = {
        'sections': SECTIONS,
        'hiddenSections': content['hiddenSections'],
        'about': content['about'],
        'experiences': map(jinjafy_experience, content['experience']),
        'education': content['education'],
        'activities': content['activities']
    }

    output = base.render(**params)

    with open(index, 'w') as outf:
        outf.write(output)

if __name__ == '__main__':
    compile_jinja()
