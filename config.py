from pathlib import Path

directory = Path(__file__).parent

YAML_FILE = directory / 'content.yaml'
JINJA_TEMPLATE = directory / 'base.jinja'
HTML_INDEX = directory / 'index.html'

SECTIONS = ['about', 'experience', 'education', 'skills', 'awards', 'activities']


FIELDS = {
    'about': [
        'tagline', 'intro'
    ],
    'experience': [
        'title', 'employer', 'start', 'end', 'description',
        'githubLink', 'customDescription'
    ],
    'education': [
        'school', 'subline', 'start', 'end',
        'currentCoursework', 'pastCoursework', 'customBody'
    ],
    'activites': [
        'icon', 'name', 'start', 'end', 'description'   
    ]
}