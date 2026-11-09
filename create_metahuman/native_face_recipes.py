"""Explicit native mouth/expression control combinations, independent of identity."""


def recipes():
    def pair(name,value):return {name+side:value for side in ('L','R')}
    def lips(name,value):return {name+side:value for side in ('UL','UR','DL','DR')}
    return [
        ('PP',{**lips('mouthLipsTogether',.8),**pair('mouthLipsPress',.6)}),
        ('FF',{'jawOpen':.12,**pair('mouthLowerLipRollIn',.55),**pair('mouthUpperLipRaise',.15)}),
        ('TH',{'jawOpen':.18,'tongueOut':.2,'tongueUp':.15}),
        ('DD',{'jawOpen':.22,'tongueUp':.35,'tongueTipUp':.6}),
        ('kk',{'jawOpen':.3,'tongueUp':.25}),
        ('CH',{'jawOpen':.2,**lips('mouthFunnel',.35)}),
        ('SS',{'jawOpen':.12,**pair('mouthCornerPull',.2)}),
        ('nn',{'jawOpen':.18,'tongueTipUp':.3}),
        ('RR',{'jawOpen':.2,**lips('mouthFunnel',.3),'tongueTipUp':.2}),
        ('aa',{'jawOpen':.8,**pair('mouthCornerWide',.1)}),
        ('E',{'jawOpen':.5,**pair('mouthCornerWide',.45)}),
        ('ih',{'jawOpen':.3,**pair('mouthCornerWide',.25)}),
        ('oh',{'jawOpen':.55,**lips('mouthFunnel',.8)}),
        ('ou',{'jawOpen':.25,**lips('mouthLipsPurse',.9)}),
        ('Blink',pair('eyeBlink',1.)),
        ('Smile',{**pair('mouthCornerPull',1.),**pair('eyeCheekRaise',.35)}),
        ('BrowsRaised',{**pair('browRaiseIn',.7),**pair('browRaiseOuter',.6)}),
        ('BrowsLowered',pair('browDown',.7)),
        ('EyesWide',pair('eyeWiden',.5)),
    ]
