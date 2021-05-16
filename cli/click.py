#!/usr/bin/env python3
import click


@click.group()
@click.argument("greetings")
def cli(greetings: str) -> str:
    click.echo(f"{greetings}, world")


@cli.command()
@cli.argument("n", type=int, help="count from 0 to n")
def repeat(n: int) -> str:
    for i in range(0, n):
        print(i)


if __name__ == "__main__":
    cli()
