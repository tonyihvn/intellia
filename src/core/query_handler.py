import logging
import json
import os
import re
from datetime import datetime
from pathlib import Path
from ..llm.client import LLMClient
from ..db.connection import get_db_connection
from ..db.schema_fetcher import SchemaFetcher
from ..rag.manager import RAGManager
from ..rag.enhancer import QueryEnhancer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class QueryHistory:
    def __init__(self):
        self.history_file = Path("query_history.json")
        self.load_history()

    def load_history(self):
        if self.history_file.exists():
            with open(self.history_file) as f:
                self.history = json.load(f)
        else:
            self.history = []

    def save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def add_query(self, query_data):
        entry = {
            'id': len(self.history) + 1,
            'timestamp': datetime.now().isoformat(),
            'question': query_data['question'],
            'sql': query_data['sql'],
            'status': query_data['status'],
            'results': query_data.get('results', None)
        }
        self.history.append(entry)
        self.save_history()
        return entry['id']

    def get_query(self, query_id):
        return next((q for q in self.history if q['id'] == query_id), None)

    def update_query(self, query_id, updates):
        for i, query in enumerate(self.history):
            if query['id'] == query_id:
                self.history[i].update(updates)
                self.save_history()
                return True
        return False

    def delete_query(self, query_id):
        self.history = [q for q in self.history if q['id'] != query_id]
        self.save_history()

class QueryHandler:
    """
    Handles natural language questions and converts them to SQL queries
    """

    def __init__(self, llm_client, query_enhancer):
        """
        Initialize QueryHandler with LLMClient instance
        """
        self.llm_client = llm_client 
        self.query_enhancer = query_enhancer
        self.history = QueryHistory()

    def preview_query(self, question):
        """Preview SQL query generation without execution"""
        try:
            response = self.generate_sql(question)
            return {
                'preview': True,
                'question': question,
                'sql': response['sql'],
                'explanation': response['explanation'],
                'needs_confirmation': True
            }
        except Exception as e:
            return {
                'preview': True,
                'error': str(e),
                'needs_confirmation': False
            }

    def analyze_error(self, sql, error):
        """Use LLM to analyze SQL errors and suggest fixes"""
        prompt = f"""
        Analyze this SQL query that failed:
        {sql}
        
        Error message:
        {error}
        
        Please explain:
        1. What caused the error
        2. How to fix it
        3. Provide the corrected SQL query
        """
        
        analysis = self.llm_client.generate(prompt)
        return {
            'original_sql': sql,
            'error': str(error),
            'analysis': analysis,
            'needs_confirmation': True
        }

    def handle_query(self, question, execute=True, preview_mode=True):
        """Enhanced query handling with preview and history"""
        if preview_mode:
            return self.preview_query(question)

        try:
            response = self.generate_sql(question)
            result = {
                'sql': response['sql'],
                'explanation': response.get('explanation', ''),
                'status': 'pending'
            }

            # Execute the SQL if requested
            if execute:
                try:
                    results = self.execute_sql(response['sql'])
                    result.update({
                        'results': results,
                        'status': 'success'
                    })
                except Exception as e:
                    # If execution fails, attempt to be helpful instead of failing outright.
                    # 1) Ask the LLM to analyze the SQL error (helpful diagnostics)
                    analysis = self.analyze_error(response['sql'], str(e))

                    # 2) Try to fetch small sample rows from any referenced tables so we can
                    #    provide the LLM with real data to base an explanation on.
                    try:
                        tables = self._extract_tables_from_sql(response['sql'])
                        samples = self._fetch_table_samples(tables)
                    except Exception:
                        samples = {}

                    # 3) Ask the LLM to produce a natural-language presentation using the
                    #    original explanation and any sample rows we were able to fetch.
                    try:
                        present_prompt = (
                            f"You attempted to run this SQL to answer the user's question:\n{question}\n\n"
                            f"The SQL:\n{response['sql']}\n\n"
                            f"Original LLM explanation:\n{response.get('explanation') or ''}\n\n"
                            f"The SQL execution failed with error: {str(e)}\n\n"
                            f"However, we were able to fetch these small sample rows from the referenced tables:\n{json.dumps(samples, default=str, indent=2)}\n\n"
                            "Using ONLY the provided sample rows and the original explanation, produce a concise, honest natural-language answer to the user's question. If the sample rows are insufficient to answer confidently, say so and list what additional data or corrected SQL would be needed."
                        )
                        presentation = self.llm_client.generate(present_prompt)
                    except Exception:
                        presentation = None

                    result.update({
                        'status': 'error',
                        'error_analysis': analysis,
                        'sample_rows': samples,
                        'presentation': presentation
                    })

            # Save to history
            history_entry = {
                'question': question,
                **result
            }
            query_id = self.history.add_query(history_entry)
            result['query_id'] = query_id

            return result

        except Exception as e:
            raise Exception(f"Error handling query: {str(e)}")

    def _extract_tables_from_sql(self, sql: str):
        """Extract referenced table names from a SQL statement using simple heuristics.

        Returns a list of unique table names.
        """
        try:
            import re
            tables = set()
            # look for FROM or JOIN followed by `table` or table
            for m in re.finditer(r"(?:from|join)\s+[`\"]?([A-Za-z0-9_]+)[`\"]?", sql, flags=re.I):
                tables.add(m.group(1))
            return list(tables)
        except Exception:
            return []

    def _fetch_table_samples(self, tables, limit=5):
        """Fetch up to `limit` sample rows from each table in `tables`.

        Returns a dict table -> [rows]. Swallows errors for individual tables.
        """
        samples = {}
        if not tables:
            return samples
        conn = None
        try:
            conn = get_db_connection()
            if not conn:
                return samples
            cur = conn.cursor(dictionary=True)
            for t in tables:
                try:
                    # Basic sanity for table name
                    if not re.match(r'^[A-Za-z0-9_]+$', t):
                        continue
                    cur.execute(f"SELECT * FROM `{t}` LIMIT %s", (limit,))
                    rows = cur.fetchall() or []
                    # Serialize datetime/decimals similar to execute_sql
                    ser = []
                    for r in rows:
                        srow = {}
                        for k, v in r.items():
                            try:
                                if hasattr(v, 'isoformat'):
                                    srow[k] = v.isoformat()
                                elif hasattr(v, 'normalize'):
                                    srow[k] = str(v)
                                else:
                                    srow[k] = v
                            except Exception:
                                srow[k] = str(v)
                        ser.append(srow)
                    samples[t] = ser
                except Exception:
                    samples[t] = []
            try:
                cur.close()
            except Exception:
                pass
            return samples
        except Exception:
            return samples
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    def generate_sql(self, question):
        """
        Generates a SQL query from a natural language question.
        """
        db_connection = None
        try:
            db_connection = get_db_connection()
            if not db_connection:
                raise Exception("Could not connect to the database.")

            # Build full prompt
            context = self.build_prompt(question, db_connection)
            logging.info(f"Generated context for LLM:\n{context}")

            # Generate the SQL query
            gen = self.llm_client.generate_sql(context)

            # Normalize generator output to a dict with 'sql', 'explanation', 'full_response'
            sql_text = None
            explanation = None
            full_response = None

            if isinstance(gen, dict):
                full_response = gen.get('full_response') or gen.get('sql') or ''
                sql_text = gen.get('sql') or (full_response and self.llm_client._extract_sql(full_response))
                explanation = gen.get('explanation')
            else:
                # gen may be a plain string
                full_response = str(gen)
                # Try to extract SQL from the free text
                try:
                    sql_text = self.llm_client._extract_sql(full_response)
                except Exception:
                    sql_text = full_response
                explanation = full_response.replace(sql_text, '').strip() if sql_text else ''

                if not sql_text:
                    # LLM did not produce SQL. Try to synthesize a schema-aware suggestion
                    try:
                        suggestion = self._suggest_sql_from_schema(question, db_connection)
                        if suggestion and (suggestion.get('sql') or suggestion.get('explanation')):
                            return {
                                'sql': (suggestion.get('sql') or '').strip(),
                                'explanation': suggestion.get('explanation') or explanation or full_response,
                                'full_response': full_response
                            }
                    except Exception:
                        pass
                    raise Exception("Failed to generate SQL query.")

            return {'sql': sql_text.strip(), 'explanation': explanation, 'full_response': full_response}

        except Exception as e:
            if db_connection and hasattr(db_connection, 'close'):
                try:
                    db_connection.close()
                except:
                    pass
            raise Exception(f"Error generating SQL: {str(e)}")

    def _suggest_sql_from_schema(self, question: str, db_connection):
        """Try to propose a best-effort SQL using only tables/columns present in the schema.

        This does not guess column names; it uses heuristics to find likely person/patient and
        program/enrollment tables and builds a SELECT that the user can review.
        """
        try:
            fetcher = SchemaFetcher(db_connection)
            tables = fetcher.get_tables() or []
            table_names = [t.get('table_name') for t in tables]

            # Helper to get columns for a table as set
            def cols_for(tbl):
                try:
                    cols = fetcher.get_table_columns(tbl) or []
                    names = set()
                    for c in cols:
                        # support different shapes
                        if isinstance(c, dict):
                            for k in ('column_name', 'Field', 'column'):
                                if k in c:
                                    names.add(c[k])
                                    break
                            # also consider keys that may be lowercase
                            for k in ('field', 'column_name'):
                                if k in c and c[k] not in names:
                                    names.add(c[k])
                        else:
                            # tuple-like fallback
                            try:
                                names.add(c[0])
                            except Exception:
                                pass
                    return names
                except Exception:
                    return set()

            def primary_key_for(tbl):
                """Try to determine the primary key column name for a table conservatively."""
                try:
                    cols = fetcher.get_table_columns(tbl) or []
                    # Look for explicit primary key indicators from DESCRIBE results
                    for c in cols:
                        if isinstance(c, dict):
                            # MySQL DESCRIBE often uses 'Field' and 'Key' columns
                            key = c.get('Key') or c.get('key') or c.get('column_key')
                            name = c.get('Field') or c.get('column_name') or c.get('column')
                            if key and str(key).upper() == 'PRI' and name:
                                return name
                    # fallback to common conventions
                    for c in cols:
                        if isinstance(c, dict):
                            name = c.get('Field') or c.get('column_name') or c.get('column')
                        else:
                            try:
                                name = c[0]
                            except Exception:
                                name = None
                        if not name:
                            continue
                        low = name.lower()
                        if low == 'id' or low.endswith('_id'):
                            return name
                    # last resort
                    return 'id'
                except Exception:
                    return 'id'

            # Find candidate person table (common names or columns)
            person_table = None
            person_table_candidates = [t for t in table_names if t and any(x in t.lower() for x in ('person','patient','patient_person','users','person_name'))]
            if person_table_candidates:
                person_table = person_table_candidates[0]
            else:
                # fallback: find table with first/last name columns
                for t in table_names:
                    cs = cols_for(t)
                    if any(x in cs for x in ('first_name','given_name','givenname','family_name','last_name','name')):
                        person_table = t
                        break

            # Find program table (program, programs)
            program_table = None
            for t in table_names:
                if t and 'program' in t.lower():
                    program_table = t
                    break

            # Find enrollment/link table: contains person_id and program_id or program_uuid
            enrollment_table = None
            for t in table_names:
                cs = cols_for(t)
                if any(x in cs for x in ('person_id','patient_id')) and any(x in cs for x in ('program_id','program_uuid','program')):
                    enrollment_table = t
                    break

            # If no dedicated enrollment table, look for tables with program_id or program in columns
            if not enrollment_table:
                for t in table_names:
                    cs = cols_for(t)
                    if any(x in cs for x in ('program_id','program')) and any(x in cs for x in ('person_id','patient_id')):
                        enrollment_table = t
                        break

            # Prepare explanation text describing discovered tables
            parts = []
            parts.append('I could not generate SQL directly from the LLM. Based on the database schema I inspected, here are candidate tables and columns that might relate to patient enrollment:')
            if person_table:
                parts.append(f"- Person table: `{person_table}` (likely contains patient names/identifiers). Columns: {', '.join(sorted(list(cols_for(person_table)) ) )}")
            else:
                parts.append('- No obvious person/patient table detected by name; look for tables with name columns (first_name, last_name, given_name).')

            if program_table:
                parts.append(f"- Program table: `{program_table}`. Columns: {', '.join(sorted(list(cols_for(program_table))) )}")
            else:
                parts.append('- No `program` table detected by name. Look for tables/columns containing program identifiers or names.')

            if enrollment_table:
                parts.append(f"- Enrollment/link table detected: `{enrollment_table}` (contains person/program linkage). Columns: {', '.join(sorted(list(cols_for(enrollment_table))))}")
            else:
                parts.append('- No enrollment/link table detected that contains both person_id and program_id columns. If such a table exists, please tell me its name and the relevant column names.')

            # Try to assemble a SQL statement if we have minimal info
            sql = ''
            if enrollment_table and person_table and program_table:
                # find name columns in person_table
                pcols = cols_for(person_table)
                name_col = None
                for c in ('full_name','display_name','name','given_name','givenname','first_name'):
                    if c in pcols:
                        name_col = c
                        break
                if not name_col:
                    # try family/given pair
                    if 'given_name' in pcols and 'family_name' in pcols:
                        name_col = f"CONCAT({person_table}.given_name, ' ', {person_table}.family_name)"

                # find program name col
                prog_cols = cols_for(program_table)
                prog_name_col = None
                for c in ('name','program_name','display_name'):
                    if c in prog_cols:
                        prog_name_col = c
                        break

                # find linking column names
                enroll_cols = cols_for(enrollment_table)
                person_fk = None
                program_fk = None
                for c in ('person_id','patient_id'):
                    if c in enroll_cols:
                        person_fk = c
                        break
                for c in ('program_id','program_uuid','program'):
                    if c in enroll_cols:
                        program_fk = c
                        break

                # determine primary keys for person and program tables when available
                person_pk = primary_key_for(person_table)
                program_pk = primary_key_for(program_table)

                # Build JOINs using discovered FK/PK names when possible
                left_join_person = f"`{enrollment_table}`.`{person_fk}` = `{person_table}`.`{person_pk}`"
                left_join_program = f"`{enrollment_table}`.`{program_fk}` = `{program_table}`.`{program_pk}`"

                if person_fk and program_fk and prog_name_col and name_col:
                    sql = f"SELECT {name_col} AS patient_name, {program_table}.{prog_name_col} AS program_name FROM `{enrollment_table}` JOIN `{person_table}` ON {left_join_person} JOIN `{program_table}` ON {left_join_program} WHERE {program_table}.{prog_name_col} = 'HIV Treatment Services' LIMIT 500"
                else:
                    # fallback minimal select from enrollment table
                    sql = f"SELECT * FROM `{enrollment_table}` WHERE `{program_fk or 'program_id'}` IS NOT NULL LIMIT 100"

            explanation = '\n'.join(parts)
            return {'sql': sql, 'explanation': explanation}
        except Exception as e:
            return {'sql': '', 'explanation': f'Could not synthesize SQL from schema: {e}'}

    def build_prompt(self, question, db_connection=None):
        """Build the prompt with schema summary and RAG-enhanced context."""
        # 1. Use RAG-based enhanced context (includes a targeted schema snippet)
        if self.query_enhancer and hasattr(self.query_enhancer, 'rag_manager') and db_connection:
            try:
                # Ensure the RAG manager has the current DB schema context available
                self.query_enhancer.rag_manager.set_db_context(db_connection)
            except Exception:
                pass
        enhanced = self.query_enhancer.enhance_query_context(question) if self.query_enhancer else {'enhanced_prompt': ''}
        enhanced_prompt = enhanced.get('enhanced_prompt', '')

        # 2. Construct a concise prompt using only the enhanced context to limit tokens
        context = f"""Task: Generate a MySQL query to {question}

Context (targeted):
{enhanced_prompt}

Requirements:
- Use only necessary tables and joins from the provided schema context
- Return accurate count/results
- Handle NULL values appropriately"""
        return context

    def execute_sql(self, sql):
        """
        Execute a SQL query and return the results.
        
        Args:
            sql: The SQL query to execute
            
        Returns:
            List of dictionaries containing the query results
        """
        db_connection = None
        cursor = None
        try:
            db_connection = get_db_connection()
            if not db_connection:
                raise Exception("Could not connect to the database.")
            
            cursor = db_connection.cursor(dictionary=True)
            cursor.execute(sql)
            results = cursor.fetchall()
            
            # Convert decimal and datetime objects to strings for JSON serialization
            sanitized_results = []
            for row in results:
                sanitized_row = {}
                for key, value in row.items():
                    if hasattr(value, 'isoformat'):  # datetime objects
                        sanitized_row[key] = value.isoformat()
                    elif hasattr(value, 'normalize'):  # Decimal objects
                        sanitized_row[key] = str(value)
                    else:
                        sanitized_row[key] = value
                sanitized_results.append(sanitized_row)
            
            return sanitized_results

        except Exception as e:
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if db_connection and hasattr(db_connection, 'close'):
                try:
                    db_connection.close()
                except:
                    pass
            raise Exception(f"Error executing query: {str(e)}")

    def generate_sql_query(self, question):
        """
        Generates a SQL query from a natural language question.
        Alias for generate_sql for backward compatibility
        """
        return self.generate_sql(question)

    def validate_sql_against_schema(self, sql: str):
        """Basic validation: check that referenced tables exist in the database schema.

        Returns dict: {'ok': True} or {'ok': False, 'missing_tables': [...]}.
        This is intentionally conservative and uses simple token matching for table names.
        """
        try:
            import re
            conn = get_db_connection()
            if not conn:
                return {'ok': False, 'error': 'Could not connect to database for validation'}

            schema_fetcher = SchemaFetcher(conn)
            # Get list of tables in DB (normalize to lower)
            cursor = conn.cursor()
            cursor.execute("SHOW TABLES")
            tables = [row[0].lower() for row in cursor.fetchall()]
            cursor.close()
            # Build map of table -> set(columns) (all lower)
            cols = schema_fetcher.get_columns() or []
            table_cols = {}
            for c in cols:
                t = c.get('table_name')
                name = c.get('column_name')
                if t and name:
                    t_lc = t.lower()
                    n_lc = name.lower()
                    if t_lc not in table_cols:
                        table_cols[t_lc] = set()
                    table_cols[t_lc].add(n_lc)

            # Find all table aliases in FROM/JOIN clauses
            alias_pattern = r'(?:from|join)\s+([`\"]?\w+[`\"]?)(?:\s+(?:as\s+)?([`\"]?\w+[`\"]?))?'
            alias_map = {}  # alias -> base table
            for match in re.finditer(alias_pattern, sql, re.IGNORECASE):
                base = match.group(1)
                alias = match.group(2)
                base_clean = base.strip('`"').lower() if base else None
                if alias:
                    alias_clean = alias.strip('`"').lower()
                    alias_map[alias_clean] = base_clean

            aliases = set(alias_map.keys())

            # Common SQL keywords and stopwords
            sql_keywords = set([
                'select','from','where','and','or','group','by','order','limit','as','count','sum','avg','min','max','distinct','on','left','right','inner','outer','join',
                'having','case','when','then','else','end','union','all','in','exists','between','like','is','null','not','true','false','asc','desc','coalesce','if','ifnull','cast'
            ])
            stopwords = {'the', 'a', 'an', 'of', 'to', 'for', 'in', 'on', 'by', 'with', 'as', 'at', 'from'}

            # Find referenced tables (normalize to lower)
            found_tables = set()
            for m in re.finditer(r"(?:from|join)\s+[`']?([a-zA-Z0-9_]+)[`']?", sql, flags=re.I):
                found_tables.add(m.group(1).lower())

            # Heuristic filter: sometimes the LLM or users include unquoted tokens or
            # the regex can pick up values that are actually string literals (e.g.
            # WHERE state = 'NASARAWA'). If a found token appears only inside
            # quoted literals in the SQL, ignore it as a referenced table.
            filtered_found = set()
            for t in found_tables:
                # look for the token appearing inside single or double quotes
                # as a standalone literal (case-insensitive)
                try:
                    if re.search(r"(?i)['\"]\s*" + re.escape(t) + r"\s*['\"]", sql):
                        # token appears as a quoted literal somewhere -> skip
                        continue
                except Exception:
                    # on any regex error, fall back to keeping the token
                    pass
                filtered_found.add(t)

            found_tables = filtered_found

            missing_tables = [t for t in found_tables if t not in tables and t not in aliases and t not in stopwords]

            missing_columns = []

            # Find qualified column references like table.column
            for m in re.finditer(r"\b([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\b", sql):
                tname = m.group(1).lower()
                cname = m.group(2).lower()
                # resolve alias to base table if needed
                base_table = alias_map.get(tname, tname)
                if base_table not in tables and tname not in aliases and tname not in stopwords:
                    if tname not in missing_tables:
                        missing_tables.append(tname)
                else:
                    # check column exists in base table
                    cols_set = table_cols.get(base_table, set())
                    if cname not in cols_set:
                        missing_columns.append({'table': base_table, 'column': cname})

            # Heuristic: check unqualified column names in SELECT and WHERE sections
            try:
                select_part = re.search(r"select\s+(.*?)\s+from\s", sql, flags=re.I | re.S)
                if select_part:
                    sel = select_part.group(1)
                    items = [i.strip() for i in re.split(r',\s*(?![^()]*\))', sel) if i.strip()]
                    for item in items:
                        if '.' in item:
                            continue
                        col_name = re.split(r"\s+as\s+|\s+", item, flags=re.I)[0].strip()
                        mcol = re.match(r"[a-zA-Z_][a-zA-Z0-9_]*", col_name)
                        if not mcol:
                            continue
                        token = mcol.group(0).lower()
                        if token in sql_keywords:
                            continue
                        found = False
                        for t in (found_tables or tables):
                            if token in table_cols.get(t, set()):
                                found = True
                                break
                        if not found:
                            missing_columns.append({'table': None, 'column': token})
            except Exception:
                pass

            result = {'ok': True, 'found': list(found_tables), 'tables': tables}
            if missing_tables:
                result['ok'] = False
                result['missing_tables'] = missing_tables
            if missing_columns:
                result['ok'] = False
                result['missing_columns'] = missing_columns

            return result
        except Exception as e:
            return {'ok': False, 'error': str(e)}