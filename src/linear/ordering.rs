use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ParameterBlockOrdering {
    group_to_elements: BTreeMap<usize, BTreeSet<String>>,
    element_to_group: HashMap<String, usize>,
}

impl ParameterBlockOrdering {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_element_to_group(&mut self, element: impl Into<String>, group: usize) -> bool {
        let element = element.into();
        if let Some(previous_group) = self.element_to_group.insert(element.clone(), group) {
            if previous_group == group {
                return true;
            }
            self.remove_from_group(&element, previous_group);
        }
        self.group_to_elements
            .entry(group)
            .or_default()
            .insert(element);
        true
    }

    pub fn clear(&mut self) {
        self.group_to_elements.clear();
        self.element_to_group.clear();
    }

    pub fn remove(&mut self, element: &str) -> bool {
        let Some(group) = self.element_to_group.remove(element) else {
            return false;
        };
        self.remove_from_group(element, group);
        true
    }

    pub fn group_id(&self, element: &str) -> Option<usize> {
        self.element_to_group.get(element).copied()
    }

    pub fn is_member(&self, element: &str) -> bool {
        self.element_to_group.contains_key(element)
    }

    pub fn group_size(&self, group: usize) -> usize {
        self.group_to_elements.get(&group).map_or(0, BTreeSet::len)
    }

    pub fn num_elements(&self) -> usize {
        self.element_to_group.len()
    }

    pub fn num_groups(&self) -> usize {
        self.group_to_elements.len()
    }

    pub fn min_nonempty_group(&self) -> Option<usize> {
        self.group_to_elements
            .first_key_value()
            .map(|(&group, _)| group)
    }

    pub fn ordered_elements(&self) -> Vec<&str> {
        self.group_to_elements
            .values()
            .flat_map(|elements| elements.iter().map(String::as_str))
            .collect()
    }

    pub(crate) fn groups(&self) -> impl Iterator<Item = (usize, &BTreeSet<String>)> {
        self.group_to_elements
            .iter()
            .map(|(&group, elements)| (group, elements))
    }

    fn remove_from_group(&mut self, element: &str, group: usize) {
        let remove_group = self
            .group_to_elements
            .get_mut(&group)
            .is_some_and(|elements| {
                elements.remove(element);
                elements.is_empty()
            });
        if remove_group {
            self.group_to_elements.remove(&group);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orders_groups_and_moves_existing_elements() {
        let mut ordering = ParameterBlockOrdering::new();
        ordering.add_element_to_group("camera_b", 1);
        ordering.add_element_to_group("point_b", 0);
        ordering.add_element_to_group("camera_a", 1);
        ordering.add_element_to_group("point_a", 0);
        ordering.add_element_to_group("point_b", 2);

        assert_eq!(
            ordering.ordered_elements(),
            ["point_a", "camera_a", "camera_b", "point_b"]
        );
        assert_eq!(ordering.group_id("point_b"), Some(2));
        assert_eq!(ordering.group_size(0), 1);
        assert_eq!(ordering.num_groups(), 3);
        assert_eq!(ordering.num_elements(), 4);

        assert!(ordering.remove("camera_a"));
        assert!(!ordering.remove("missing"));
        assert!(!ordering.is_member("camera_a"));
    }
}
