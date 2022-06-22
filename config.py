YAML_FILE = 'content.yaml'
JINJA_TEMPLATE = 'base.jinja'
HTML_INDEX = 'index.html'

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