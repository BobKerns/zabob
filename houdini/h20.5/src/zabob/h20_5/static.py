'''
Routines to extract static data from Houdini 20.5.
'''

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import click

import hou

from zabob.common import (
    ZABOB_OUT_DIR, OptionalType, analysis_db, analysis_db_writer, do_all, get_stored_modules,
)
from zabob.common.analyze_node_types import do_analysis


IGNORE_MODULES: Mapping[str, str] = MappingProxyType({
    'opnode_sum': "A script, not a module",
    'perfmon_sum': "A script, not a module.",
    "pycparser._build_tables": "A script that writes files.",
    "PIL.PyAccess": "Creates files.",
    "hutil.pbkdf2": "Creates files.",
    "antigravity": "Runs external program (osascript on macOS).",
    "pip.__pip-runner__": "A Script",
    "idlelib.idle": "Runs IDLE and hangs.",
    **{k: "Crashes hython 20.5"
       for k in (
                'dashbox.ui', 'dashbox.textedit', 'dashbox.common', 'dashbox',
                'generateHDAToolsForOTL', 'test.autotest', 'test.tf_inherit_check',
                'test._test_embed_structseq', 'ocio.editor',
                'hrecipes.models', 'hrecipes.manager', 'pdgd.datalayerserver',
                'assettools', 'searchbox',
                'searchbox.panetabs', 'searchbox.paths', 'searchbox.ui',
                'searchbox.categories', 'searchbox.parms', 'searchbox.tools',
                'searchbox.preferences', 'searchbox.hotkeys', 'searchbox.common',
                'searchbox.viewport_settings', 'searchbox.solaris', 'searchbox.radialmenus',
                'searchbox.help', 'searchbox.expression', 'layout.view', 'layout.assetgallery',
                'layout.panel', 'layout.brushpanel', 'stagemanager.panel',
                'shibokensupport.signature.parser', "apex.transientconstraint",
            )
       }
    })


default_db = ZABOB_OUT_DIR / hou.applicationVersionString() / 'houdini_static_data.db'
@click.command()
@click.argument('db', type=OptionalType(click.Path(exists=False, dir_okay=False, path_type=Path)), default=default_db)
def load_data(db: Path|None):
    """
    Main function to save Houdini static data to a database.
    Args:
        db (Path): The path to the SQLite database file.
    """
    if db == Path():
        db = default_db
    db = db or default_db
    db.parent.mkdir(parents=True, exist_ok=True)
    with analysis_db(db_path=db,
                     write=True,
                     ) as conn:
        with analysis_db_writer(connection=conn, label="Write") as writer:
            # Get the list of modules already stored in the database so we can skip them.
            done = set(get_stored_modules(connection=conn))
            do_all((item
                    for item in map(writer,
                                    do_analysis(connection=conn,
                                                done=done,
                                                ignore=IGNORE_MODULES))),
                   label="Analyze Houdini 20.5 static data",)
    print(f"Static data saved to {db}")


if __name__ == "__main__":
    load_data()
