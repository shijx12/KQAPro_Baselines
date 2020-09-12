from utils.value_class import ValueClass, comp
import json
from collections import defaultdict
from datetime import date
from queue import Queue

"""
For convenience of implementation, in this rule-based execution engine,
all locating functions (including And Or) return (entity_ids, facts), 
even though some of them do not concern facts.
So that we can always use `entity_ids, _ = dependencies[0]` to store outputs.
"""
constrains = {                          # dependencies, inputs, returns, function
    # functions for locating entities
    'FindAll': [0, 0],                  # []; []; [(entity_ids, facts)]; get all ids of entities and concepts 
    'Find': [0, 1],                     # []; [entity_name]; [(entity_ids, facts)]; get ids for the given name
    'FilterConcept': [1, 1],            # [entity_ids]; [concept_name]; [(entity_ids, facts)]; filter entities by concept
    'FilterStr': [1, 2],                # [entity_ids]; [key, value]; [(entity_ids, facts)]
    'FilterNum': [1, 3],                # [entity_ids]; [key, value, op]; [(entity_ids, facts)]; op should be '=','>','<', or '!='
    'FilterYear': [1, 3],               # [entity_ids]; [key, value, op]; [(entity_ids, facts)]
    'FilterDate': [1, 3],               # [entity_ids]; [key, value, op]; [(entity_ids, facts)]
    'QFilterStr': [1, 2],               # [(entity_ids, facts)]; [qualifier_key, qualifier_value]; [(entity_ids, facts)]; filter by facts
    'QFilterNum': [1, 3],               # [(entity_ids, facts)]; [qualifier_key, qualifier_value, op]; [(entity_ids, facts)];
    'QFilterYear': [1, 3],              # [(entity_ids, facts)]; [qualifier_key, qualifier_value, op]; [(entity_ids, facts)];
    'QFilterDate': [1, 3],              # [(entity_ids, facts)]; [qualifier_key, qualifier_value, op]; [(entity_ids, facts)];
    'Relate': [1, 2],                   # [entity_ids]; [predicate, direction]; [(entity_ids, facts)]; entity number should be 1
    
    # functions for logic
    'And': [2, 0],                      # [entity_ids_1, entity_ids_2]; []; [(entity_ids, facts)], intersection
    'Or': [2, 0],                       # [entity_ids_1, entity_ids_2]; []; [(entity_ids, facts)], union

    # functions for query
    'What': [1, 0],                     # [entity_ids]; []; [entity_name]; get its name, entity number should be 1
    'Count': [1, 0],                    # [entity_ids]; []; [count]
    'SelectBetween': [2, 2],            # [entity_ids_1, entity_ids_2]; [key, op]; [entity_name]; op is 'greater' or 'less', entity number should be 1
    'SelectAmong': [1, 2],              # [entity_ids]; [key, op]; [entity_name]; op is 'largest' or 'smallest'
    'QueryAttr': [1, 1],                # [entity_ids]; [key]; [value]; get the attribute value of given attribute key, entity number should be 1
    'QueryAttrUnderCondition': [1, 3],  # [entity_ids]; [key, qualifier_key, qualifier_value]; [value]; entity number should be 1
    'VerifyStr': [1, 1],                # [value]; [value]; [bool]; check whether the dependency equal to the input
    'VerifyNum': [1, 2],                # [value]; [value, op]; [bool];
    'VerifyYear': [1, 2],               # [value]; [value, op]; [bool];
    'VerifyDate': [1, 2],               # [value]; [value, op]; [bool];
    'QueryRelation': [2, 0],            # [entity_ids_1, entity_ids_2]; []; [predicate]; get the predicate between two entities, entity number should be 1
    'QueryAttrQualifier': [1, 3],       # [entity_ids]; [key, value, qualifier_key]; [qualifier_value]; get the qualifier value of the given attribute fact, entity number should be 1
    'QueryRelationQualifier': [2, 2],   # [entity_ids_1, entity_ids_2]; [predicate, qualifier_key]; [qualifier_value]; get the qualifier value of the given relation fact, entity number should be 1
}


class RuleExecutor(object):
    def __init__(self, vocab, kb_json):
        self.vocab = vocab
        print('load kb')
        kb = json.load(open(kb_json))
        self.concepts = kb['concepts']
        self.entities = kb['entities']

        # replace adjacent space and tab in name
        for con_id, con_info in self.concepts.items():
            con_info['name'] = ' '.join(con_info['name'].split())
        for ent_id, ent_info in self.entities.items():
            ent_info['name'] = ' '.join(ent_info['name'].split())

        self.entity_name_to_ids = defaultdict(list)
        for ent_id, ent_info in self.entities.items():
            self.entity_name_to_ids[ent_info['name']].append(ent_id)
        self.concept_name_to_ids = defaultdict(list)
        for con_id, con_info in self.concepts.items():
            self.concept_name_to_ids[con_info['name']].append(con_id)

        self.concept_to_entity = defaultdict(set)
        for ent_id in self.entities:
            for c in self._get_all_concepts(ent_id): # merge entity into ancestor concepts
                self.concept_to_entity[c].add(ent_id)
        self.concept_to_entity = { k:list(v) for k,v in self.concept_to_entity.items() }

        self.key_type = {}
        for ent_id, ent_info in self.entities.items():
            for attr_info in ent_info['attributes']:
                self.key_type[attr_info['key']] = attr_info['value']['type']
                for qk in attr_info['qualifiers']:
                    for qv in attr_info['qualifiers'][qk]:
                        self.key_type[qk] = qv['type']
        for ent_id, ent_info in self.entities.items():
            for rel_info in ent_info['relations']:
                for qk in rel_info['qualifiers']:
                    for qv in rel_info['qualifiers'][qk]:
                        self.key_type[qk] = qv['type']
        # Note: key_type is one of string/quantity/date, but date means the key may have values of type year
        self.key_type = { k:v if v!='year' else 'date' for k,v in self.key_type.items() }

        # parse values into ValueClass object
        for ent_id, ent_info in self.entities.items():
            for attr_info in ent_info['attributes']:
                attr_info['value'] = self._parse_value(attr_info['value'])
                for qk, qvs in attr_info['qualifiers'].items():
                    attr_info['qualifiers'][qk] = [self._parse_value(qv) for qv in qvs]
        for ent_id, ent_info in self.entities.items():
            for rel_info in ent_info['relations']:
                for qk, qvs in rel_info['qualifiers'].items():
                    rel_info['qualifiers'][qk] = [self._parse_value(qv) for qv in qvs]

        # some entities may have relations with concepts, we add them into self.concepts for visiting convenience
        for ent_id in self.entities:
            for rel_info in self.entities[ent_id]['relations']:
                obj_id = rel_info['object']
                if obj_id in self.concepts:
                    if 'relations' not in self.concepts[obj_id]:
                        self.concepts[obj_id]['relations'] = []
                    self.concepts[obj_id]['relations'].append({
                        'predicate': rel_info['predicate'],
                        'direction': 'forward' if rel_info['direction']=='backward' else 'backward',
                        'object': ent_id,
                        'qualifiers': rel_info['qualifiers'],
                        })

    def _parse_value(self, value):
        if value['type'] == 'date':
            x = value['value']
            p1, p2 = x.find('/'), x.rfind('/')
            y, m, d = int(x[:p1]), int(x[p1+1:p2]), int(x[p2+1:])
            result = ValueClass('date', date(y, m, d))
        elif value['type'] == 'year':
            result = ValueClass('year', value['value'])
        elif value['type'] == 'string':
            result = ValueClass('string', value['value'])
        elif value['type'] == 'quantity':
            result = ValueClass('quantity', value['value'], value['unit'])
        else:
            raise Exception('unsupport value type')
        return result

    def _get_direct_concepts(self, ent_id):
        """
        return the direct concept id of given entity/concept
        """
        if ent_id in self.entities:
            return self.entities[ent_id]['instanceOf']
        elif ent_id in self.concepts:
            return self.concepts[ent_id]['instanceOf']

    def _get_all_concepts(self, ent_id):
        """
        return a concept id list
        """
        ancestors = []
        q = Queue()
        for c in self._get_direct_concepts(ent_id):
            q.put(c)
        while not q.empty():
            con_id = q.get()
            ancestors.append(con_id)
            for c in self.concepts[con_id]['instanceOf']:
                q.put(c)
        return ancestors


    def forward(self, program, inputs, 
                ignore_error=False, show_details=False):
        memory = []
        program = [self.vocab['function_idx_to_token'][p] for p in program]
        inputs = [[self.vocab['word_idx_to_token'][i] for i in inp] for inp in inputs]

        try:
            # infer the dependency based on the function definition
            dependency = []
            branch_stack = []
            for i, p in enumerate(program):
                if p in {'<START>', '<END>', '<PAD>'}:
                    dep = [0, 0]
                elif p in {'FindAll', 'Find'}:
                    dep = [0, 0]
                    branch_stack.append(i - 1)
                elif p in {'And', 'Or', 'SelectBetween', 'QueryRelation', 'QueryRelationQualifier'}:
                    dep = [branch_stack[-1], i-1]
                    branch_stack = branch_stack[:-1]
                else:
                    dep = [i-1, 0]
                dependency.append(dep)

            for p, dep, inp in zip(program, dependency, inputs):
                if p == '<START>':
                    res = None
                elif p == '<END>':
                    break
                else:
                    func = getattr(self, p)
                    res = func([memory[d] for d in dep], inp)
                memory.append(res)
                if show_details:
                    print(p, dep, inp)
                    print(res)
            return str(memory[-1])
        except Exception as e:
            if ignore_error:
                return None
            else:
                raise




    def _parse_key_value(self, key, value, typ=None):
        if typ is None:
            typ = self.key_type[key]
        if typ=='string':
            value = ValueClass('string', value)
        elif typ=='quantity':
            if ' ' in value:
                vs = value.split()
                v = vs[0]
                unit = ' '.join(vs[1:])
            else:
                v = value
                unit = '1'
            value = ValueClass('quantity', float(v), unit)
        else:
            if '/' in value or ('-' in value and '-' != value[0]):
                split_char = '/' if '/' in value else '-'
                p1, p2 = value.find(split_char), value.rfind(split_char)
                y, m, d = int(value[:p1]), int(value[p1+1:p2]), int(value[p2+1:])
                value = ValueClass('date', date(y, m, d))
            else:
                value = ValueClass('year', int(value))
        return value

    def FindAll(self, dependencies, inputs):
        entity_ids = list(self.entities.keys())
        return (entity_ids, None)

    def Find(self, dependencies, inputs):
        name = inputs[0]
        entity_ids = self.entity_name_to_ids[name]
        if name in self.concept_name_to_ids: # concept may appear in some relations
            entity_ids += self.concept_name_to_ids[name]
        return (entity_ids, None)

    def FilterConcept(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        concept_name = inputs[0]
        concept_ids = self.concept_name_to_ids[concept_name]
        entity_ids_2 = []
        for i in concept_ids:
            entity_ids_2 += self.concept_to_entity[i]
        entity_ids = list(set(entity_ids) & set(entity_ids_2))
        return (entity_ids, None)

    def _filter_attribute(self, entity_ids, tgt_key, tgt_value, op, typ):
        tgt_value = self._parse_key_value(tgt_key, tgt_value, typ)
        res_ids = []
        res_facts = []
        for i in entity_ids:
            for attr_info in self.entities[i]['attributes']:
                k, v = attr_info['key'], attr_info['value']
                if k==tgt_key and v.can_compare(tgt_value) and comp(v, tgt_value, op):
                    res_ids.append(i)
                    res_facts.append(attr_info)
        return (res_ids, res_facts)

    def FilterStr(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        key, value, op = inputs[0], inputs[1], '='
        return self._filter_attribute(entity_ids, key, value, op, 'string')

    def FilterNum(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        key, value, op = inputs[0], inputs[1], inputs[2]
        return self._filter_attribute(entity_ids, key, value, op, 'quantity')

    def FilterYear(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        key, value, op = inputs[0], inputs[1], inputs[2]
        return self._filter_attribute(entity_ids, key, value, op, 'year')

    def FilterDate(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        key, value, op = inputs[0], inputs[1], inputs[2]
        return self._filter_attribute(entity_ids, key, value, op, 'date')

    def _filter_qualifier(self, entity_ids, facts, tgt_key, tgt_value, op, typ):
        tgt_value = self._parse_key_value(tgt_key, tgt_value, typ)
        res_ids = []
        res_facts = []
        for i, f in zip(entity_ids, facts):
            for qk, qvs in f['qualifiers'].items():
                if qk == tgt_key:
                    for qv in qvs:
                        if qv.can_compare(tgt_value) and comp(qv, tgt_value, op):
                            res_ids.append(i)
                            res_facts.append(f)
        return (res_ids, res_facts)

    def QFilterStr(self, dependencies, inputs):
        entity_ids, facts = dependencies[0]
        key, value, op = inputs[0], inputs[1], '='
        return self._filter_qualifier(entity_ids, facts, key, value, op, 'string')

    def QFilterNum(self, dependencies, inputs):
        entity_ids, facts = dependencies[0]
        key, value, op = inputs[0], inputs[1], inputs[2]
        return self._filter_qualifier(entity_ids, facts, key, value, op, 'quantity')

    def QFilterYear(self, dependencies, inputs):
        entity_ids, facts = dependencies[0]
        key, value, op = inputs[0], inputs[1], inputs[2]
        return self._filter_qualifier(entity_ids, facts, key, value, op, 'year')

    def QFilterDate(self, dependencies, inputs):
        entity_ids, facts = dependencies[0]
        key, value, op = inputs[0], inputs[1], inputs[2]
        return self._filter_qualifier(entity_ids, facts, key, value, op, 'date')

    def Relate(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        entity_id = entity_ids[0]
        predicate, direction = inputs[0], inputs[1]
        res_ids = []
        res_facts = []
        if entity_id in self.entities:
            rel_infos = self.entities[entity_id]['relations']
        else:
            rel_infos = self.concepts[entity_id]['relations']
        for rel_info in rel_infos:
            if rel_info['predicate']==predicate and rel_info['direction']==direction:
                res_ids.append(rel_info['object'])
                res_facts.append(rel_info)
        return (res_ids, res_facts)

    def And(self, dependencies, inputs):
        entity_ids_1, _ = dependencies[0]
        entity_ids_2, _ = dependencies[1]
        return (list(set(entity_ids_1) & set(entity_ids_2)), None)

    def Or(self, dependencies, inputs):
        entity_ids_1, _ = dependencies[0]
        entity_ids_2, _ = dependencies[1]
        return (list(set(entity_ids_1) | set(entity_ids_2)), None)

    def What(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        entity_id = entity_ids[0]
        name = self.entities[entity_id]['name']
        return name

    def Count(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        return len(entity_ids)

    def SelectBetween(self, dependencies, inputs):
        entity_ids_1, _ = dependencies[0]
        entity_ids_2, _ = dependencies[1]
        entity_id_1 = entity_ids_1[0]
        entity_id_2 = entity_ids_2[0]
        key, op = inputs[0], inputs[1]
        for attr_info in self.entities[entity_id_1]['attributes']:
            if key == attr_info['key']:
                v1 = attr_info['value']
        for attr_info in self.entities[entity_id_2]['attributes']:
            if key == attr_info['key']:
                v2 = attr_info['value']
        i = entity_id_1 if ((op=='greater' and v1>v2) or (op=='less' and v1<v2)) else entity_id_2
        name = self.entities[i]['name']
        return name

    def SelectAmong(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        key, op = inputs[0], inputs[1]
        candidates = []
        for i in entity_ids:
            for attr_info in self.entities[i]['attributes']:
                if key == attr_info['key']:
                    v = attr_info['value']
            candidates.append((i, v))
        sort = sorted(candidates, key=lambda x: x[1])
        i = sort[0][0] if op=='smallest' else sort[-1][0]
        name = self.entities[i]['name']
        return name

    def QueryAttr(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        entity_id = entity_ids[0]
        key = inputs[0]
        for attr_info in self.entities[entity_id]['attributes']:
            if key == attr_info['key']:
                v = attr_info['value']
        return v

    def QueryAttrUnderCondition(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        entity_id = entity_ids[0]
        key, qual_key, qual_value = inputs[0], inputs[1], inputs[2]
        qual_value = self._parse_key_value(qual_key, qual_value)
        for attr_info in self.entities[entity_id]['attributes']:
            if key == attr_info['key']:
                flag = False
                for qk, qvs in attr_info['qualifiers'].items():
                    if qk == qual_key:
                        for qv in qvs:
                            if qv.can_compare(qual_value) and comp(qv, qual_value, "="):
                                flag = True
                                break
                    if flag:
                        break
                if flag:
                    v = attr_info['value']
                    break
        return v

    def _verify(self, dependencies, value, op, typ):
        attr_value = dependencies[0]
        value = self._parse_key_value(None, value, typ)
        if attr_value.can_compare(value) and comp(attr_value, value, op):
            answer = 'yes'
        else:
            answer = 'no'
        return answer

    def VerifyStr(self, dependencies, inputs):
        value, op = inputs[0], '='
        return self._verify(dependencies, value, op, 'string')
        

    def VerifyNum(self, dependencies, inputs):
        value, op = inputs[0], inputs[1]
        return self._verify(dependencies, value, op, 'quantity')

    def VerifyYear(self, dependencies, inputs):
        value, op = inputs[0], inputs[1]
        return self._verify(dependencies, value, op, 'year')

    def VerifyDate(self, dependencies, inputs):
        value, op = inputs[0], inputs[1]
        return self._verify(dependencies, value, op, 'date')

    def QueryRelation(self, dependencies, inputs):
        entity_ids_1, _ = dependencies[0]
        entity_ids_2, _ = dependencies[1]
        entity_id_1 = entity_ids_1[0]
        entity_id_2 = entity_ids_2[0]
        if entity_id_1 in self.entities:
            rel_infos = self.entities[entity_id_1]['relations']
        else:
            rel_infos = self.concepts[entity_id_1]['relations']
        p = None
        for rel_info in rel_infos:
            if rel_info['object']==entity_id_2 and rel_info['direction']=='forward':
                p = rel_info['predicate']
        return p

    def QueryAttrQualifier(self, dependencies, inputs):
        entity_ids, _ = dependencies[0]
        entity_id = entity_ids[0]
        key, value, qual_key = inputs[0], inputs[1], inputs[2]
        value = self._parse_key_value(key, value)
        for attr_info in self.entities[entity_id]['attributes']:
            if attr_info['key']==key and attr_info['value'].can_compare(value) and \
                comp(attr_info['value'], value, '='):
                for qk, qvs in attr_info['qualifiers'].items():
                    if qk == qual_key:
                        return qvs[0]
        return None

    def QueryRelationQualifier(self, dependencies, inputs):
        entity_ids_1, _ = dependencies[0]
        entity_ids_2, _ = dependencies[1]
        entity_id_1 = entity_ids_1[0]
        entity_id_2 = entity_ids_2[0]
        predicate, qual_key = inputs[0], inputs[1]
        if entity_id_1 in self.entities:
            rel_infos = self.entities[entity_id_1]['relations']
        else:
            rel_infos = self.concepts[entity_id_1]['relations']
        for rel_info in rel_infos:
            if rel_info['object']==entity_id_2 and rel_info['direction']=='forward' and \
                rel_info['predicate']==predicate:
                for qk, qvs in rel_info['qualifiers'].items():
                    if qk == qual_key:
                        return qvs[0]
        return None
