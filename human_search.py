"""
For questions about valiation of quantitative attributes,
like "Is You Don't Mess with the Zohan longer than 2700 seconds?",
the answer should be "no" if the real unit is different from the asked one,
because different units cannot compare.
"""

from value_class import ValueClass, comp
import json
from collections import defaultdict
from datetime import date
from queue import Queue


class SearchEngine(object):
    def __init__(self, kb_json):
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

        self.attr_keys = set()
        self.predicates = set()
        self.units = set()
        # parse values into ValueClass object
        for ent_id, ent_info in self.entities.items():
            for attr_info in ent_info['attributes']:
                self.attr_keys.add(attr_info['key'])
                if 'unit' in attr_info['value']:
                    self.units.add(attr_info['value']['unit'])
                attr_info['value'] = self._parse_value(attr_info['value'])
                for qk, qvs in attr_info['qualifiers'].items():
                    self.attr_keys.add(qk)
                    attr_info['qualifiers'][qk] = [self._parse_value(qv) for qv in qvs]
        for ent_id, ent_info in self.entities.items():
            for rel_info in ent_info['relations']:
                self.predicates.add(rel_info['predicate'])
                for qk, qvs in rel_info['qualifiers'].items():
                    self.attr_keys.add(qk)
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



    def _parse_key_value(self, key, value, unit, typ=None):
        if typ=='string':
            value = ValueClass('string', value)
        elif typ=='quantity':
            if unit == '':
                unit = '1'
            value = ValueClass('quantity', float(value), unit)
        else:
            if '/' in value or ('-' in value and '-' != value[0]):
                split_char = '/' if '/' in value else '-'
                p1, p2 = value.find(split_char), value.rfind(split_char)
                y, m, d = int(value[:p1]), int(value[p1+1:p2]), int(value[p2+1:])
                value = ValueClass('date', date(y, m, d))
            else:
                value = ValueClass('year', int(value))
        return value


    def get_name(self, id):
        if id in self.entities:
            return self.entities[id]['name']
        else:
            return self.concepts[id]['name']

    def print_all_info(self, id):
        print('\n' + '='*50)
        if id in self.concepts:
            info = self.concepts[id]
            print("concept id: {}, name: {}".format(id, info['name']))
        else:
            info = self.entities[id]
            print("entity id: {}, name: {}".format(id, info['name']))
        anc = ["{}:{}".format(i, self.concepts[i]['name']) for i in self._get_all_concepts(id)]
        print("ancestor concepts: {}".format(";  ".join(anc)))
        for attr_info in info.get('attributes', []):
            print('  > {}: {}'.format(attr_info['key'], attr_info['value']))
            for qk, qvs in attr_info['qualifiers'].items():
                print('     - {}: {}'.format(qk, '; '.join([str(qv) for qv in qvs])))
        for rel_info in info.get('relations', []):
            print('  > {} ({}): {}:{}'.format(rel_info['predicate'], 
                rel_info['direction'], rel_info['object'], self.get_name(rel_info['object'])))
            for qk, qvs in rel_info['qualifiers'].items():
                print('     - {}: {}'.format(qk, '; '.join([str(qv) for qv in qvs])))

    def by_name(self, name):
        results = self.entity_name_to_ids[name]
        if name in self.concept_name_to_ids: # concept may appear in some relations
            results += self.concept_name_to_ids[name]
        print('===== match with {} ====='.format(name))
        for i in results:
            print(' - {}:{}'.format(i, self.get_name(i)))
        for i in results:
            self.print_all_info(i)

    def part_name(self, name):
        results = []
        for i in self.concepts:
            if name in self.concepts[i]['name']:
                results.append(i)
        for i in self.entities:
            if name in self.entities[i]['name']:
                results.append(i)
        print('===== partial match with {} ====='.format(name))
        for i in results:
            print(' - {}:{}'.format(i, self.get_name(i)))

    def part_attr(self, x):
        results = []
        for a in self.attr_keys:
            if x in a:
                results.append(a)
        print('===== partial match with {} ====='.format(x))
        for i in results:
            print(' - {}'.format(i))

    def part_pred(self, x):
        results = []
        for a in self.predicates:
            if x in a:
                results.append(a)
        print('===== partial match with {} ====='.format(x))
        for i in results:
            print(' - {}'.format(i))

    def part_unit(self, x):
        results = []
        for a in self.units:
            if x in a:
                results.append(a)
        print('===== partial match with {} ====='.format(x))
        for i in results:
            print(' - {}'.format(i))


    def filter_concept(self, con_name):
        concept_ids = self.concept_name_to_ids[con_name]
        results = []
        for i in concept_ids:
            results += self.concept_to_entity[i]
        results = set(results)
        print('===== entities of concept {} ====='.format(con_name))
        for i in results:
            print(' - {}:{}'.format(i, self.get_name(i)))
        return results

    def filter_attr(self, tgt_key, tgt_value, tgt_unit, op, typ):
        tgt_value = self._parse_key_value(tgt_key, tgt_value, tgt_unit, typ)
        res_ids = []
        res_facts = []
        for i in self.entities:
            for attr_info in self.entities[i]['attributes']:
                k, v = attr_info['key'], attr_info['value']
                if k==tgt_key and v.can_compare(tgt_value) and comp(v, tgt_value, op):
                    res_ids.append(i)
                    res_facts.append(attr_info)
        print('===== entities that satisfy the attribute condition =====')
        for i, attr_info in zip(res_ids ,res_facts):
            print(' - {}:{}'.format(i, self.get_name(i)))
            for qk, qvs in attr_info['qualifiers'].items():
                print('     - {}: {}'.format(qk, '; '.join([str(qv) for qv in qvs])))
        return (res_ids, res_facts)


    def filter_qualifier(self, entity_ids, facts, tgt_key, tgt_value, tgt_unit, op, typ):
        tgt_value = self._parse_key_value(tgt_key, tgt_value, tgt_unit, typ)
        res_ids = []
        res_facts = []
        for i, f in zip(entity_ids, facts):
            for qk, qvs in f['qualifiers'].items():
                if qk == tgt_key:
                    for qv in qvs:
                        if qv.can_compare(tgt_value) and comp(qv, tgt_value, op):
                            res_ids.append(i)
                            res_facts.append(f)
        print('===== remained entities =====')
        for i in set(res_ids):
            print(' - {}:{}'.format(i, self.get_name(i)))
        return (res_ids, res_facts)


    def filter_pred(self, entity_id, predicate, direction):
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
        print('===== entities that satisfy the realtion condition =====')
        for i in set(res_ids):
            print(' - {}:{}'.format(i, self.get_name(i)))
        return (res_ids, res_facts)

    def find_rel(self, ent_1, ent_2):
        if ent_1 in self.entities:
            rel_infos = self.entities[ent_1]['relations']
        else:
            rel_infos = self.concepts[ent_1]['relations']
        for rel_info in rel_infos:
            if rel_info['object'] == ent_2:
                print('{}:{} {} {}:{} ({})'.format(ent_1, self.get_name(ent_1), rel_info['predicate'],
                    ent_2, self.get_name(ent_2), rel_info['direction']))
                for qk, qvs in rel_info['qualifiers'].items():
                    print('     - {}: {}'.format(qk, '; '.join([str(qv) for qv in qvs])))

    def show_attr(self, entity_ids, key):
        for i in entity_ids:
            print(' - {}:{}'.format(i, self.get_name(i)))
            for attr_info in self.entities[i]['attributes']:
                if key != attr_info['key']:
                    continue
                print('  > {}: {}'.format(attr_info['key'], attr_info['value']))
                for qk, qvs in attr_info['qualifiers'].items():
                    print('     - {}: {}'.format(qk, '; '.join([str(qv) for qv in qvs])))
        



if __name__ == '__main__':
    eng = SearchEngine('./test_dataset/kb.json')
    from IPython import embed
    embed()
