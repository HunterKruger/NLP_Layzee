import json
import pandas as pd

class Header:
    def __init__(self, names: list, types: list):
        self.names = names
        self.types = types

    def __getitem__(self, idx):
        return self.names[idx], self.types[idx]

    def __len__(self):
        return len(self.names)

    def __repr__(self):
        return ' | '.join(['{}({})'.format(n, t) for n, t in zip(self.names, self.types)])

class Table:
    def __init__(self, id, name, title, header: Header, rows, **kwargs):
        self.id = id
        self.name = name
        self.title = title
        self.header = header
        self.rows = rows
        self._df = None

    @property
    def df(self):
        if self._df is None:
            self._df = pd.DataFrame(data=self.rows,
                                    columns=self.header.names,
                                    dtype=str)
        return self._df

    def _repr_html_(self):
        return self.df._repr_html_()

class Tables:
    table_dict = None

    def __init__(self, table_list: list = None, table_dict: dict = None):
        self.table_dict = {}
        if isinstance(table_list, list):
            for table in table_list:
                self.table_dict[table.id] = table
        if isinstance(table_dict, dict):
            self.table_dict.update(table_dict)

    def push(self, table):
        self.table_dict[table.id] = table

    def __len__(self):
        return len(self.table_dict)

    def __add__(self, other):
        return Tables(
            table_list=list(self.table_dict.values()) +
            list(other.table_dict.values())
        )

    def __getitem__(self, id):
        return self.table_dict[id]

    def __iter__(self):
        for table_id, table in self.table_dict.items():
            yield table_id, table

class Question:
    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return self.text

    def __getitem__(self, idx):
        return self.text[idx]

    def __len__(self):
        return len(self.text)

class SQL:
    op_sql_dict = {0: ">", 1: "<", 2: "==", 3: "!="}
    agg_sql_dict = {0: "", 1: "AVG", 2: "MAX", 3: "MIN", 4: "COUNT", 5: "SUM"}
    conn_sql_dict = {0: "NULL", 1: "AND", 2: "OR"}

    def __init__(self, cond_conn_op: int, agg: list, sel: list, conds: list, **kwargs):
        self.cond_conn_op = cond_conn_op
        self.sel = []
        self.agg = []
        sel_agg_pairs = zip(sel, agg)
        sel_agg_pairs = sorted(sel_agg_pairs, key=lambda x: x[0])
        for col_id, agg_op in sel_agg_pairs:
            self.sel.append(col_id)
            self.agg.append(agg_op)
        self.conds = sorted(conds, key=lambda x: x[0])

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def keys(self):
        return ['cond_conn_op', 'sel', 'agg', 'conds']

    def __getitem__(self, key):
        return getattr(self, key)

    def to_json(self):
        return json.dumps(dict(self), ensure_ascii=False, sort_keys=True)

    def equal_all_mode(self, other):
        return self.to_json() == other.to_json()

    def equal_agg_mode(self, other):
        self_sql = SQL(cond_conn_op=0, agg=self.agg, sel=self.sel, conds=[])
        other_sql = SQL(cond_conn_op=0, agg=other.agg, sel=other.sel, conds=[])
        return self_sql.to_json() == other_sql.to_json()

    def equal_conn_and_agg_mode(self, other):
        self_sql = SQL(cond_conn_op=self.cond_conn_op,
                       agg=self.agg,
                       sel=self.sel,
                       conds=[])
        other_sql = SQL(cond_conn_op=other.cond_conn_op,
                        agg=other.agg,
                        sel=other.sel,
                        conds=[])
        return self_sql.to_json() == other_sql.to_json()

    def equal_no_val_mode(self, other):
        self_sql = SQL(cond_conn_op=self.cond_conn_op,
                       agg=self.agg,
                       sel=self.sel,
                       conds=[cond[:2] for cond in self.conds])
        other_sql = SQL(cond_conn_op=other.cond_conn_op,
                        agg=other.agg,
                        sel=other.sel,
                        conds=[cond[:2] for cond in other.conds])
        return self_sql.to_json() == other_sql.to_json()

    def __eq__(self, other):
        raise NotImplementedError('compare mode not set')

    def __repr__(self):
        repr_str = ''
        repr_str += "sel: {}\n".format(self.sel)
        repr_str += "agg: {}\n".format([self.agg_sql_dict[a]
                                        for a in self.agg])
        repr_str += "cond_conn_op: '{}'\n".format(
            self.conn_sql_dict[self.cond_conn_op])
        repr_str += "conds: {}".format(
            [[cond[0], self.op_sql_dict[cond[1]], cond[2]] for cond in self.conds])

        return repr_str

    def _repr_html_(self):
        return self.__repr__().replace('\n', '<br>')

class Query:
    def __init__(self, question: Question, table: Table, sql: SQL = None):
        self.question = question
        self.table = table
        self.sql = sql

    def _repr_html_(self):
        repr_str = '{}<br>{}<br>{}'.format(
            self.table._repr_html_(),
            self.question.__repr__(),
            self.sql._repr_html_() if self.sql is not None else ''
        )
        return repr_str

def read_tables(table_file):
    tables = Tables()
    with open(table_file, encoding='utf-8') as f:
        for line in f:
            tb = json.loads(line)
            header = Header(tb.pop('header'), tb.pop('types'))
            table = Table(header=header, **tb)
            tables.push(table)
    return tables

def read_data(data_file, tables: Tables):
    queries = []
    with open(data_file, encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            question = Question(text=data['question'])
            table = tables[data['table_id']]
            if 'sql' in data:
                sql = SQL.from_dict(data['sql'])
            else:
                sql = None
            query = Query(question=question, table=table, sql=sql)
            queries.append(query)
    return queries
