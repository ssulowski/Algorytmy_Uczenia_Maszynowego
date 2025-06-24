import adsk.core, adsk.fusion, traceback
import random
import os
import csv
import math

app = adsk.core.Application.get()
ui = app.userInterface



def run(context):
    try:

        design = adsk.fusion.Design.cast(app.activeProduct)
        rootComp = design.rootComponent

        # Ścieżka zapisu
        output_folder = r"C:\Users\szymi\OneDrive\Pulpit\Studia\PROJEKT_FIGURKI\Nożyki\Testowe_Obrot"
        os.makedirs(output_folder, exist_ok=True)

        # CSV
        csv_path = os.path.join(output_folder, 'model_data.csv')
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'Model_ID', 'Length', 'Width', 'Height', 'Cut_Count', 'Depth',
            'X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2', 'X3', 'Y3', 'Z3'
        ])

        # Parametry
        number_of_models = 1
        min_cuts = 5
        max_cuts = 12

        for model_index in range(number_of_models):
            # Usuń istniejące ciała i szkice
            while rootComp.bRepBodies.count > 0:
                rootComp.bRepBodies.item(0).deleteMe()
            while rootComp.sketches.count > 0:
                rootComp.sketches.item(0).deleteMe()

            # Tworzenie szkicu i bryły głównej
            sketch = rootComp.sketches.add(rootComp.xYConstructionPlane)

            length = random.uniform(2.5, 7.5)
            width = random.uniform(1.5, 4.5)
            height = random.uniform(0.5, 2.0)

            p1 = adsk.core.Point3D.create(0, 0, 0)
            p2 = adsk.core.Point3D.create(length, width, 0)
            sketch.sketchCurves.sketchLines.addTwoPointRectangle(p1, p2)

            prof = sketch.profiles.item(0)
            extrudes = rootComp.features.extrudeFeatures
            ext_input = extrudes.createInput(prof, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
            ext_input.setDistanceExtent(False, adsk.core.ValueInput.createByReal(height))
            ext = extrudes.add(ext_input)

            main_body = ext.bodies.item(0)
            if main_body is None:
                return

            num_cuts = random.randint(min_cuts, max_cuts)

            def find_circle_center(p1, p2, p3):
                # Wydobywamy współrzędne YZ
                y1, z1 = p1.y, p1.z
                y2, z2 = p2.y, p2.z
                y3, z3 = p3.y, p3.z

                # Wzory analityczne na środek okręgu opisanego na 3 punktach w 2D
                A = y1 * (z2 - z3) - y2 * (z1 - z3) + y3 * (z1 - z2)
                if abs(A) < 1e-6:
                    return None  # Punkty współliniowe, nie ma okręgu

                B = (y1**2 + z1**2) * (z3 - z2) + (y2**2 + z2**2) * (z1 - z3) + (y3**2 + z3**2) * (z2 - z1)
                C = (y1**2 + z1**2) * (y2 - y3) + (y2**2 + z2**2) * (y3 - y1) + (y3**2 + z3**2) * (y1 - y2)

                y_center = -B / (2 * A)
                z_center = -C / (2 * A)

                # Zakładamy, że X jest taki sam jak pozostałych punktów (leżą na płaszczyźnie YZ)
                x_center = p1.x

                return adsk.core.Point3D.create(x_center, y_center, z_center)

            def rotate_around_z(pt, center, angle):
                x = pt.x - center.x
                y = pt.y - center.y
                x_new = x * math.cos(angle) - y * math.sin(angle)
                y_new = x * math.sin(angle) + y * math.cos(angle)
                return adsk.core.Point3D.create(x_new + center.x, y_new + center.y, pt.z)
            
            for i in range(num_cuts):
                # Punkty łuku
                rot_deg = random.uniform(-20, 20)
                rot_rad = math.radians(rot_deg)
                num_points = random.randint(3, 10)
                x_start = random.uniform(0.2, length - 0.5)
                x_end = x_start + random.uniform(0.2, 0.5)                
                y = width/2
                max_depth = max(abs((x_start - x_end)/2), height*0.3)
                depth_range = max_depth
                max_z = random.uniform(height - 0.05, height - depth_range)
                deepest_z = max_z
                mid_x = x_start

                points = []

                for i in range(num_points):
                    t = i/(num_points - 1)
                    x = x_start + t*(x_end - x_start)

                    if i == num_points//2:
                        z = max_z
                        deepest_z = z
                        mid_x = x
                    elif i == 0 or i == num_points - 1:
                        z = height
                    else:
                        prev_z = points[i-1].z
                        if i < num_points//2:
                            max_step = (prev_z - max_z) * 0.2
                            delta = prev_z - random.uniform(0.01, max(0.01, max_step))
                            z = max(delta, max_z)
                        else:
                            max_step = (height - prev_z) * 0.2
                            delta = prev_z + random.uniform(0.01, max(0.01, max_step))
                            z = min(delta, height)
                    
                    points.append(adsk.core.Point3D.create(x,y,z))

                margin = random.uniform(0.1, 0.4)
                center_point = adsk.core.Point3D.create(mid_x, y, deepest_z)
                start_point =  adsk.core.Point3D.create(mid_x, 0 + margin, height)
                end_point = adsk.core.Point3D.create(mid_x, width - margin, height)
                circle_center_point = find_circle_center(start_point, center_point, end_point)
                radius = circle_center_point.z - deepest_z

                rotated_points = [rotate_around_z(pt, circle_center_point, rot_rad) for pt in points]
                points_collection = adsk.core.ObjectCollection.create()
                for pt in rotated_points:
                    points_collection.add(pt)

                spline_sketch = rootComp.sketches.add(rootComp.xYConstructionPlane)
                splines = spline_sketch.sketchCurves.sketchFittedSplines
                spline = splines.add(points_collection)


                axis_start_pre = adsk.core.Point3D.create(circle_center_point.x - radius, circle_center_point.y, circle_center_point.z)
                axis_end_pre = adsk.core.Point3D.create(circle_center_point.x + radius, circle_center_point.y, circle_center_point.z)
                axis_start = rotate_around_z(axis_start_pre, circle_center_point, rot_rad)
                axis_end   = rotate_around_z(axis_end_pre,   circle_center_point, rot_rad)

                start_pt = spline.fitPoints.item(0).geometry
                end_pt = spline.fitPoints.item(spline.fitPoints.count - 1).geometry
                start1 = adsk.core.Point3D.create(start_pt.x, start_pt.y, circle_center_point.z)
                end1 = adsk.core.Point3D.create(end_pt.x, end_pt.y, circle_center_point.z)
                
                lines = spline_sketch.sketchCurves.sketchLines
                lines.addByTwoPoints(start_pt, start1)
                lines.addByTwoPoints(end_pt, end1)
                lines.addByTwoPoints(start1, end1)

                axis_sketch = rootComp.sketches.add(rootComp.xYConstructionPlane)
                axis_line = axis_sketch.sketchCurves.sketchLines.addByTwoPoints(axis_start, axis_end)

                

                prof = None
                for p in spline_sketch.profiles:
                    if p.areaProperties().area > 0:
                        prof = p
                        break
                revolve_input = rootComp.features.revolveFeatures.createInput(prof, axis_line, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
                angle = adsk.core.ValueInput.createByString('360 deg')
                revolve_input.setAngleExtent(False, angle)
                revolve = rootComp.features.revolveFeatures.add(revolve_input)
                revolved_body = revolve.bodies.item(0)

                try:
                    tool_bodies = adsk.core.ObjectCollection.create()
                    tool_bodies.add(revolved_body)
                    combine_input = rootComp.features.combineFeatures.createInput(main_body, tool_bodies)
                    combine_input.operation = adsk.fusion.FeatureOperations.CutFeatureOperation
                    combine_input.isKeepToolBodies = False
                    combine_input.isNewComponent = False
                    rootComp.features.combineFeatures.add(combine_input)
                except:
                    num_cuts += 1
                      
                
                for i in range(len(points_collection)):
                    if i == 0:
                        point1 = points_collection[i]
                    if i == num_points//2:
                        point2 = points_collection[i]
                        maximum_depth = height - points_collection[i].z
                    if i == num_points-1:
                        point3 = points_collection[i] 

                csv_writer.writerow([
                    model_index + 1,  # Model_ID
                    round(length, 3), round(width, 3), round(height, 3), num_cuts,                    
                    round(maximum_depth, 3),round(point1.x, 3), round(point1.y, 3), round(point1.z, 3),
                    round(point2.x, 3), round(point2.y, 3), round(point2.z, 3),
                    round(point3.x, 3), round(point3.y, 3), round(point3.z, 3)
                ])

            # Eksport STL
            exportMgr = design.exportManager
            filename = os.path.join(output_folder, f'model_{model_index + 1}.stl')
            stl_options = exportMgr.createSTLExportOptions(main_body, filename)
            stl_options.meshRefinement = adsk.fusion.MeshRefinementSettings.MeshRefinementMedium
            exportMgr.execute(stl_options)

        csv_file.close()
        ui.messageBox(f'Wygenerowano {number_of_models} modeli. Dane zapisano w CSV.')

    except Exception as e:
        if ui:
            ui.messageBox(f'Błąd: {str(e)}\n{traceback.format_exc()}')
