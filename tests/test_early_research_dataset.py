"""Synthetic unit fixtures and an offline research writer integration check."""
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from f1_predictor.early.policy import CANDIDATE_FEATURES, EXCLUDED_FEATURES
from f1_predictor.early.research_features import ResearchBatch, ResearchRequest, load_research_tables
from f1_predictor.early.research_dataset import DATASET_SCHEMA, apply_calendar_corrections, attach_outcomes, validation_block, validate_schedule, write_dataset
from f1_predictor.stage3.sources import SourceStore


class ResearchDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = SourceStore(Path(self.temp.name) / 'sources')
        self.identity = dict(race_id=1080, year=2023, round=1, grand_prix_id='bahrain', circuit_id='bahrain', session='event')
        self.request = ResearchRequest(1080, '2023-03-03T11:30:00+00:00', '2023-03-02T11:30:00+00:00', (), 'test-fixture', 'Not historical evidence', True, (), {})
        frame = pd.DataFrame([{'driver_id': f'driver-{i}', 'constructor_id': 'team', **{k:v for k,v in self.identity.items() if k!='session'}} for i in range(1,7)])
        self.batch = ResearchBatch(frame, [], (), self.request)
        self.results = pd.DataFrame([dict(raceId=1080, driverId=f'driver-{i}', constructorId='team', positionNumber=i if i<=3 else np.nan,
            positionDisplayOrder=i, positionText=str(i) if i<=3 else {4:'DNF',5:'DSQ',6:'DNS'}[i], points={1:25,2:18,3:15}.get(i,np.nan)) for i in range(1,7)])

    def source(self, kind, published='2023-03-01T12:00:00+00:00', pairs=None, **scope):
        ref=self.store.capture('f1','https://www.formula1.com/unit-fixture/'+kind,b'Synthetic test evidence; not a race source',
            document_kind=kind,published_at=published,publication_basis='document',reviewed_by='test-fixture')
        return self.store.review(ref, applicability={**self.identity,'driver_constructor_pairs':pairs or [],**scope},
            reviewed_by='test-fixture', notes='Technical unit fixture, not historical proof')

    def exception(self, **changes):
        ref=self.source('withdrawal','2023-03-03T12:00:00+00:00',pairs=[['driver-7','team']])
        return dict(driver_id='driver-7',constructor_id='team',reference=ref,sha256=self.store.read(ref)[0]['sha256'],reason='Test-only late withdrawal',**changes)

    def test_labels_distinguish_classification_order_and_nonstart(self):
        labels,coverage=attach_outcomes(self.batch,self.results,self.store)
        self.assertEqual(labels.race_winner.tolist(),[1,0,0,0,0,0])
        self.assertEqual(labels.podium_finish.tolist(),[1,1,1,0,0,0])
        self.assertEqual(labels.points_finish.tolist(),[1,1,1,0,0,0])
        self.assertEqual(labels.finish_order.iloc[3:5].tolist(),[4,5])
        self.assertTrue(pd.isna(labels.finish_order.iloc[5]))
        self.assertEqual(coverage['ranked_entries'],5)

    def test_points_are_official_points_not_top_ten(self):
        results=self.results.copy()
        results.loc[0,'points']=0
        labels,_=attach_outcomes(self.batch,results,self.store)
        self.assertEqual(labels.loc[0,'points_finish'],0)
        self.assertEqual(labels.loc[0,'race_winner'],1)

    def test_single_saturday_practice_keeps_friday_cutoff(self):
        from f1_predictor.early.policy import scheduled_cutoff
        request = replace(self.request, race_id=1031, fp1_start='2020-10-31T09:00:00+00:00',
                          cutoff=scheduled_cutoff('2020-10-31T09:00:00+00:00'))
        self.assertEqual(request.cutoff, '2020-10-30T09:00:00+00:00')
        tables = {'races': pd.DataFrame([dict(id=1031, year=2020, round=13,
            grandPrixId='emilia-romagna', circuitId='imola', date='2020-11-01')])}
        ref = self.source('event_schedule', race_id=1031, year=2020, round=13,
            grand_prix_id='emilia-romagna', circuit_id='imola', fp1_start=request.fp1_start,
            race_start='2020-11-01T12:10:00+00:00', evidence_mode='retrospective_research')
        packet = dict(request=asdict(request), schedule=dict(reference=ref, sha256=self.store.read(ref)[0]['sha256']))
        validate_schedule(packet, tables, self.store)
        packet['request']['fp1_start'] = '2020-10-30T09:00:00+00:00'
        with self.assertRaisesRegex(ValueError, 'FP1 differs'):
            validate_schedule(packet, tables, self.store)

    def test_half_points_remain_positive_and_do_not_change_order(self):
        results=self.results.copy()
        results['points']/=2
        original,_=attach_outcomes(self.batch,self.results,self.store)
        labels,_=attach_outcomes(self.batch,results,self.store)
        pd.testing.assert_frame_equal(labels,original)

    def test_calendar_overlay_is_reviewed_scoped_nonmutating_and_research_only(self):
        tables={'races':pd.DataFrame([dict(id=1080,year=2023,round=1,grandPrixId='bahrain',circuitId='bahrain',date='2023-02-26')])}
        original=tables['races'].copy()
        ref=self.source('event_schedule',fp1_start=self.request.fp1_start,race_start='2023-03-05T15:00:00+00:00',evidence_mode='retrospective_research')
        correction=dict(race_id=1080,original_date='2023-02-26',reference=ref,sha256=self.store.read(ref)[0]['sha256'])
        corrected=apply_calendar_corrections(tables,[correction],self.store)
        pd.testing.assert_frame_equal(tables['races'],original)
        self.assertEqual(corrected['races'].iloc[0].date,'2023-03-05')
        packet=dict(request=asdict(self.request),schedule=dict(reference=ref,sha256=correction['sha256']))
        with self.assertRaisesRegex(ValueError,'Race date differs'):
            validate_schedule(packet,tables,self.store)
        validate_schedule(packet,corrected,self.store)
        for changes in ({'sha256':'0'*64},{'original_date':'2023-02-25'},{'race_id':999},{'extra':True}):
            with self.assertRaises(ValueError):apply_calendar_corrections(tables,[{**correction,**changes}],self.store)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            apply_calendar_corrections(tables,[correction,correction],self.store)
        future={'races':tables['races'].assign(year=2024)}
        with self.assertRaisesRegex(ValueError,'research source year'):
            apply_calendar_corrections(future,[correction],self.store)

    def test_calendar_overlay_rejects_results_wrong_scope_and_unknown_timezone(self):
        tables={'races':pd.DataFrame([dict(id=1080,year=2023,round=1,grandPrixId='bahrain',circuitId='bahrain',date='2023-02-26')])}
        for kind,scope in [('race_results',{}),('event_schedule',{'race_id':1081}),('event_schedule',{'race_start':'2023-03-05T15:00:00'}),('event_schedule',{'evidence_mode':'prospective'})]:
            values=dict(fp1_start=self.request.fp1_start,race_start='2023-03-05T15:00:00+00:00',evidence_mode='retrospective_research')
            values.update(scope)
            ref=self.source(kind,**values)
            correction=dict(race_id=1080,original_date='2023-02-26',reference=ref,sha256=self.store.read(ref)[0]['sha256'])
            with self.assertRaises(ValueError):apply_calendar_corrections(tables,[correction],self.store)

    def test_missing_result_requires_independent_withdrawal_review(self):
        self.batch.frame.loc[6]={**self.batch.frame.iloc[0].to_dict(),'driver_id':'driver-7'}
        with self.assertRaisesRegex(ValueError,'Unexplained missing'):
            attach_outcomes(self.batch,self.results,self.store)
        labels,coverage=attach_outcomes(self.batch,self.results,self.store,[self.exception()])
        self.assertEqual(len(labels),7)
        self.assertTrue(pd.isna(labels.finish_order.iloc[-1]))
        self.assertEqual(labels.iloc[-1].outcome_source,'reviewed_withdrawal')
        self.assertEqual(coverage['reviewed_missing_result_drivers'],['driver-7'])

    def test_future_substitute_is_reported_but_not_inserted(self):
        results=pd.concat([self.results,pd.DataFrame([dict(raceId=1080,driverId='reserve',constructorId='team',positionNumber=4,positionDisplayOrder=7,positionText='4',points=12)])],ignore_index=True)
        labels,coverage=attach_outcomes(self.batch,results,self.store)
        self.assertEqual(len(labels),6)
        self.assertEqual(coverage['result_drivers_outside_announced_field'],['reserve'])
        self.assertEqual(coverage['outside_field_binary_positives']['points_finish'],1)
        self.assertFalse(coverage['target_coverage_complete']['points_finish'])
        self.assertTrue(coverage['target_coverage_complete']['race_winner'])

    def test_date_only_withdrawal_must_be_unambiguously_after_cutoff(self):
        self.batch.frame.loc[6]={**self.batch.frame.iloc[0].to_dict(),'driver_id':'driver-7'}
        for day, accepted in [('2023-03-02',False),('2023-03-03',False),('2023-03-04',True)]:
            ref=self.store.capture('f1','https://www.formula1.com/unit-fixture/date-withdrawal',
                b'Synthetic date-only withdrawal',document_kind='withdrawal')
            ref=self.store.review(ref,applicability={**self.identity,'driver_constructor_pairs':[['driver-7','team']],
                'research_publication_date':dict(date=day,basis='document',timezone='unknown')},
                reviewed_by='test-fixture',notes='Synthetic date-only research claim')
            exception=dict(driver_id='driver-7',constructor_id='team',reference=ref,
                sha256=self.store.read(ref)[0]['sha256'],reason='Synthetic late withdrawal')
            with self.subTest(day=day):
                if not accepted:
                    with self.assertRaisesRegex(ValueError,'withdrawal evidence'):
                        attach_outcomes(self.batch,self.results,self.store,[exception])
                else:
                    labels,_=attach_outcomes(self.batch,self.results,self.store,[exception])
                    self.assertEqual(labels.iloc[-1].outcome_source,'reviewed_withdrawal')
                    self.assertIsNone(self.store.read(ref)[0]['published_at'])

    def test_unannounced_winner_cannot_be_hidden_by_announced_field_metrics(self):
        results=self.results.copy()
        results['positionNumber']+=1
        results['positionDisplayOrder']+=1
        for index in range(3):results.loc[index,'positionText']=str(index+2)
        results=pd.concat([results,pd.DataFrame([dict(raceId=1080,driverId='reserve',constructorId='team',positionNumber=1,positionDisplayOrder=1,positionText='1',points=25)])],ignore_index=True)
        labels,coverage=attach_outcomes(self.batch,results,self.store)
        self.assertEqual(labels.race_winner.sum(),0)
        self.assertEqual(coverage['outside_field_binary_positives']['race_winner'],1)
        self.assertFalse(coverage['target_coverage_complete']['race_winner'])

    def test_invalid_duplicate_or_incomplete_classifications_fail(self):
        cases=[pd.concat([self.results,self.results.iloc[[0]]]),self.results.iloc[1:].copy()]
        for key,value in [('positionDisplayOrder',1),('positionNumber',2.5),('points',-1)]:
            changed=self.results.copy();changed.loc[3,key]=value;cases.append(changed)
        for results in cases:
            with self.assertRaises(ValueError): attach_outcomes(self.batch,results,self.store)

    def test_wrong_team_and_nonstart_with_points_fail(self):
        for key,value in [('constructorId','other'),('points',1),('positionNumber',6)]:
            results=self.results.copy();results.loc[5,key]=value
            with self.assertRaises(ValueError): attach_outcomes(self.batch,results,self.store)

    def test_withdrawal_hash_scope_and_unused_exception_fail(self):
        with self.assertRaisesRegex(ValueError,'Unused'):
            attach_outcomes(self.batch,self.results,self.store,[self.exception()])
        self.batch.frame.loc[6]={**self.batch.frame.iloc[0].to_dict(),'driver_id':'driver-7'}
        bad=self.exception();bad['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'withdrawal evidence'):
            attach_outcomes(self.batch,self.results,self.store,[bad])
        ref=self.source('withdrawal','2023-03-03T12:00:00+00:00',pairs=[['driver-7','team']],race_id=1081)
        bad.update(reference=ref,sha256=self.store.read(ref)[0]['sha256'])
        with self.assertRaisesRegex(ValueError,'Wrong event'):
            attach_outcomes(self.batch,self.results,self.store,[bad])

    def test_other_race_results_cannot_change_labels(self):
        changed=self.results.assign(raceId=1081)
        a,_=attach_outcomes(self.batch,self.results,self.store)
        b,_=attach_outcomes(self.batch,pd.concat([self.results,changed]),self.store)
        pd.testing.assert_frame_equal(a,b)

    def test_block_membership_does_not_slide_for_partial_cohorts(self):
        races=pd.DataFrame({'year':[2023]*22,'id':range(100,122),'round':range(1,23),'date':pd.date_range('2023-01-01',periods=22,freq='7D')})
        self.assertEqual([validation_block(i,races) for i in (107,108,114,115,121)],['selection','calibration','calibration','report','report'])
        with self.assertRaisesRegex(ValueError,'22-race'):
            validation_block(115,races.iloc[1:])

    def test_offline_writer_is_deterministic_private_portable_and_separates_targets(self):
        # Three reviewed fixture identities, not a claim about complete race entries.
        people=[('max-verstappen','red-bull','honda-rbpt'),('sergio-perez','red-bull','honda-rbpt'),('fernando-alonso','aston-martin','mercedes')]
        roster=self.source('season_registration',pairs=[[d,t] for d,t,e in people])
        request=replace(self.request,entries=tuple(dict(driver_id=d,constructor_id=t,engine_manufacturer_id=e,evidence=[roster]) for d,t,e in people),source_hashes={roster:self.store.read(roster)[0]['sha256']})
        schedule=self.source('event_schedule',fp1_start=request.fp1_start,evidence_mode='retrospective_research')
        packet=dict(request=asdict(request),schedule=dict(reference=schedule,sha256=self.store.read(schedule)[0]['sha256']))
        historical=self.source('event_schedule',race_id=1051,year=2021,round=16,grand_prix_id='turkey',circuit_id='istanbul',
            fp1_start='2021-10-08T08:30:00+00:00',race_start='2021-10-10T12:00:00+00:00',evidence_mode='retrospective_research')
        correction=dict(race_id=1051,original_date='2021-10-03',reference=historical,sha256=self.store.read(historical)[0]['sha256'])
        path=Path(self.temp.name)/'packets.json'
        path.write_text(json.dumps(dict(schema_version=DATASET_SCHEMA,evidence_mode='retrospective_research',packets=[packet],calendar_corrections=[correction])))
        output=Path(self.temp.name)/'dataset'
        manifest=write_dataset(ROOT,path,self.store,output)
        repeated=Path(self.temp.name)/'repeated'
        write_dataset(ROOT,output/'packets.json',SourceStore(output/'sources'),repeated)
        for name in ('features.csv','outcomes.csv','coverage.json','missingness.json'):
            self.assertEqual((output/name).read_bytes(),(repeated/name).read_bytes())
        features=pd.read_csv(output/'features.csv')
        outcomes=pd.read_csv(output/'outcomes.csv')
        self.assertEqual(features.columns.tolist()[4:],list(CANDIDATE_FEATURES))
        self.assertFalse(EXCLUDED_FEATURES & set(features.columns))
        self.assertFalse({'race_winner','finish_order'} & set(features.columns))
        self.assertEqual(outcomes.race_winner.sum(),1)
        self.assertFalse(manifest['point_in_time_verified'])
        self.assertFalse(manifest['training_roster_coverage_complete'])
        self.assertEqual(manifest['blocks'],dict(train=0,selection=1,calibration=0,report=0))
        self.assertEqual(manifest['calendar_corrections'],[correction])
        self.assertEqual(SourceStore(output/'sources').read(historical)[0]['sha256'],correction['sha256'])
        for name,digest in manifest['files_sha256'].items():
            self.assertEqual(hashlib.sha256((output/name).read_bytes()).hexdigest(),digest)
        with self.assertRaises(FileExistsError): write_dataset(ROOT,path,self.store,output)
        tables=load_research_tables(ROOT/'data/raw')
        self.assertEqual(tables['races'].set_index('id').loc[1051,'date'],'2021-10-03')
        bad=copy.deepcopy(packet);bad['request']['fp1_start']='2023-03-03T12:30:00+00:00'
        with self.assertRaisesRegex(ValueError,'FP1 differs'): validate_schedule(bad,tables,self.store)
        bad=copy.deepcopy(packet);bad['schedule']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'Schedule hash'): validate_schedule(bad,tables,self.store)


if __name__=='__main__': unittest.main()
