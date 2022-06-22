import io
import pprint

from typing import Iterable

import click
from ruamel.yaml import YAML

from compile_jinja import compile_jinja
from config import YAML_FILE, SECTIONS, FIELDS


yaml = YAML()


def blank_fields(section: str):
    """Return a YAML-like string containing the blank fields for the given section."""
    prefix = ' - '  # so YAML knows to start the list
    indent = ' ' * 3  # indent each field three spaces for readability
    output = []
    for i, field in enumerate(FIELDS[section]):
        prepend = prefix if i == 0 else indent
        output.append(prepend + field + ': ')
    # add newline at the end to avoid disrupting subsequent file contents
    return '\n'.join(output) + '\n'


@click.group()
def actions():
    pass


@actions.result_callback()
def callback(result, **kwargs):
    compile_jinja()


@actions.command()
@click.argument('section', type=click.Choice(SECTIONS, case_sensitive=False))
def new(section):
    # get the section information from the user
    new_entry = click.edit(blank_fields(section), extension='.yaml')
    # open the raw content.yaml file
    with open('content.yaml', 'r') as f:
        lines = f.readlines()
    # get the line number in the yaml file where the given section begins
    section_start = -1
    for i, line in enumerate(lines):
        if section in line:
            section_start = i
    # insert the information after the line where the section begins
    lines.insert(section_start + 1, new_entry)
    # write to file
    with open(YAML_FILE, 'w') as f:
        f.writelines(lines)


def display_dict(d):
    GLOBAL_INDENT = 2
    output = []
    for k, v in d.items():
        k_output = f'{k}: '
        formatted_v = pprint.pformat(v)
        # split on newline and add the correct indentation
        indent = GLOBAL_INDENT + len(k_output)
        joiner = '\n' + (indent * ' ')  # replace all newlines with newline + indentation
        v_output = joiner.join(formatted_v.split('\n'))
        entry_output = (GLOBAL_INDENT * ' ') + k_output + v_output 
        output.append(entry_output)
    return '\n'.join(output)


def yaml_to_string(yaml_obj) -> str:
    f = io.StringIO()
    yaml.dump(dict(yaml_obj), f)
    f.seek(0)
    return f.read()


@actions.command()
@click.argument('section', type=click.Choice(SECTIONS, case_sensitive=False))
def edit(section):
    # read in yaml for given section
    with open('content.yaml', 'r') as f:
        raw_text = f.read()
        data = yaml.load(raw_text)

    multiple_entries = False
    # if there's only one entry for the section, find it
    if isinstance(data[section], dict):
        to_edit = yaml_to_string(data[section])

    # if there's multiple entries, prompt user for which they'd like to edit
    else:
        multiple_entries = True
        click.echo(f'Which entry would you like to edit?')
        # display the entries
        # certain sections have lists of repeated items, whereas others
        # just have one dict of kv pairs
        # if this section just has one dict:
        entries = data[section]
        for i, entry in enumerate(entries):
            click.secho(f'{i} ------', fg='cyan')
            click.echo(display_dict(entry))
        # create iterable of valid choices
        # note: convert ints to str bc click prompt treats input as strings
        choices = [str(i) for i in range(len(entries))]

        idx_selected = click.prompt(
            click.style('Enter an integer to select an entry:', fg='yellow'),
            prompt_suffix=' ',
            type=click.Choice(choices),
            show_choices=False
            )
        # click prompt returns a str no matter what
        idx_selected = int(idx_selected)
        to_edit = yaml_to_string(data[section][idx_selected])

    # get user to edit the entry
    edited = click.edit(to_edit, extension='.yaml')

    # if the user quit without saving
    if edited is None:
        click.secho('Edit aborted.', fg='red')
        return

    # replace old with edited entry
    edited_yaml = yaml.load(edited)

    to_update = data[section]
    if multiple_entries:
        to_update = to_update[idx_selected]
    to_update.update(edited_yaml)

    with open(YAML_FILE, 'w') as f:
        yaml.dump(data, f)
    
    click.secho('Change successful!', fg=40) # bright green, see 8-bit terminal colors
    

@actions.command()
@click.argument('section', type=click.Choice(SECTIONS, case_sensitive=False))
def delete(section):
    # read in yaml for given section
    with open('content.yaml', 'r') as f:
        raw_text = f.read()
        data = yaml.load(raw_text)
    multiple_entries = False
    # if there's only one entry for the section just use it
    confirmation_display = data[section]
    # if there are multiple entries for the section, prompt the user
    # to choose which to delete
    if isinstance(data[section], list):
        multiple_entries = True
        click.echo(f'Which entry would you like to delete?')
        for i, entry in enumerate(data[section]):
            click.secho(f'{i} ------', fg='cyan')
            click.echo(display_dict(entry))
        
        # create iterable of valid choices
        # note: convert ints to str bc click prompt treats input as strings
        choices = [str(i) for i in range(len(data[section]))]

        idx_selected = click.prompt(
            click.style('Enter an integer to select an entry:', fg='yellow'),
            prompt_suffix=' ',
            type=click.Choice(choices),
            show_choices=False
            )
        
        idx_selected = int(idx_selected)
        # update confirmation_display with selected choice
        confirmation_display = data[section][int(idx_selected)]

    # then, make the user confirm deletion
    click.secho('Do you really want to delete the following item?', fg='red')
    click.echo(display_dict(confirmation_display))
    try:
        answer = click.confirm(click.style("Type 'y' to proceed", fg='bright_red'))
    except click.Abort:
        pass

    if not answer:
        click.echo('Delete aborted.')
        return

    # delete the entries from the yaml object
    if multiple_entries:
        data[section].pop(idx_selected)
    else:
        data.pop(section)
    
    with open(YAML_FILE, 'w') as f:
        yaml.dump(data, f)
    
    click.secho('Change successful!', fg=40) # bright green, see 8-bit terminal colors    


def prettify(strs: Iterable[str]):
    output = []
    for s in strs:
        output.append(' - ' + s)
    return '\n'.join(output)


@actions.command()
@click.option('-s', '--section', required=False)
def ls(section):
    """List the sections."""
    if section:
        info_msg = f'Required fields for section {section}: \n'
        output = FIELDS[section]
    else:
        info_msg = f'Available sections: \n'
        output = SECTIONS
    click.echo(info_msg + prettify(output))

if __name__ == '__main__':
    actions()