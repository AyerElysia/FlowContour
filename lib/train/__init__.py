"""Shared training engine and objectives for the three mainline stages.

Entry points import the concrete trainer, optimizer and recorder modules
directly. Keeping this package initializer side-effect free lets reward and
policy tests run without parsing command-line training arguments.
"""
