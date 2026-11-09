"""Bounded native endpoint search; every probe restores the artist's original state."""
import numpy as np
from . import native_surface_commit as native


def quality(report,fraction):
    validation=report.get('surface_validation',{});alignment=validation.get('source_alignment',{});distance=validation.get('surface_distance_upper_bounds',{})
    rms=sum(distance.get(direction,{}).get('rms_m',0.) for direction in ('forward','reverse'))
    return (alignment.get('after_back_faces',0),report['crossings_after'],alignment.get('after_severe_reference_area_m2',0.),alignment.get('after_energy',0.),rms,-fraction)


def commit(context,target,name,output,*,reference,validator=None,retractor=None,property_updates=None,refinement=3):
    if not 0<=refinement<=3:raise ValueError('Native partial-step refinement is bounded to three probes')
    output=np.asarray(output,dtype=float);reference=np.asarray(reference,dtype=float);attempts=[];last_error=None
    def attempt(fraction,publish):
        planned=reference+(output-reference)*fraction;retraction=None
        if fraction<1 and retractor:planned,retraction=retractor(planned)
        try:
            try:report=native.commit(context,target,name,planned,reference=reference,validator=validator,property_updates=property_updates,publish=publish)
            except ValueError as exc:
                if retraction is not None or not retractor or 'source-detail bounds' not in str(exc):raise
                planned,retraction=retractor(planned)
                report=native.commit(context,target,name,planned,reference=reference,validator=validator,property_updates=property_updates,publish=publish)
            report['step_fraction']=fraction
            if retraction:report['endpoint_retraction']=retraction
            attempts.append(dict(fraction=fraction,valid=True,published=publish,quality=list(quality(report,fraction))))
            return planned,report
        except ValueError as exc:
            attempts.append(dict(fraction=fraction,valid=False,published=False,reason=str(exc)));raise
    high=1.;best=None
    for fraction in (1.,.5,.25,.125,.0625,.03125):
        try:
            planned,report=attempt(fraction,publish=fraction==1.)
            if fraction==1.:
                report['endpoint_search']=dict(method='BOUNDED_NATIVE_PARTIAL_STEP_V1',attempts=attempts,refinement_probes=0);return report
            best=(quality(report,fraction),fraction,planned,report);low=fraction;break
        except ValueError as exc:last_error=exc;high=fraction
    if best is None:raise last_error
    for _ in range(refinement):
        fraction=(low+high)*.5
        try:
            planned,report=attempt(fraction,publish=False);low=fraction
            candidate=(quality(report,fraction),fraction,planned,report)
            if candidate[0]<best[0]:best=candidate
        except ValueError:high=fraction
    _,fraction,planned,probe=best
    # Publish from the unchanged original state and rerun every check. Probe
    # results never authorize bypassing the commit boundary or its source caps.
    report=native.commit(context,target,name,planned,reference=reference,validator=validator,property_updates=property_updates)
    report['step_fraction']=fraction
    if probe.get('endpoint_retraction'):report['endpoint_retraction']=probe['endpoint_retraction']
    report['endpoint_search']=dict(method='BOUNDED_NATIVE_PARTIAL_STEP_V1',attempts=attempts,refinement_probes=refinement,selection='BACK_FACES_CONTACTS_SEVERE_AREA_ALIGNMENT_DISTANCE',selected_quality=list(quality(report,fraction)),not_a_global_step_optimum=True)
    return report
