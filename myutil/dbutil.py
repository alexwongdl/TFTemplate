"""
Created by Alex.W.
On 2017-08-07
"""

from contextlib import contextmanager

import pymysql


@contextmanager
def connect(**kwargs):
    con = None
    try:
        con = pymysql.connect(**kwargs)
        yield con
    finally:
        if con:
            con.close()


@contextmanager
def cursor(conn):
    cur = None
    try:
        cur = conn.cursor()
        yield cur
    finally:
        if cur:
            cur.close()


@contextmanager
def get_cursor(**conn_kwargs):
    with connect(**conn_kwargs) as conn:
        with cursor(conn) as cur:
            yield cur


@contextmanager
def get_connect_cursor(**conn_kwargs):
    with connect(**conn_kwargs) as conn:
        with cursor(conn) as cur:
            yield conn, cur
