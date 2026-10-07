"""CLI command modules, one per command or group of related commands.

Nothing is imported here: each module pulls in what its command needs, and a
run of one command should not pay for the others. The command functions are
reached through their own modules or, lazily, through deepecgkit.cli.
"""
